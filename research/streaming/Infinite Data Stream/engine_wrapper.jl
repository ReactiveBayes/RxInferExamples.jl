module InfiniteDataStreamEngineWrapper

using RxInfer, Rocket

export EngineWithFEStream, create_engine_with_fe_stream

"""
    EngineWithFEStream

A wrapper around RxInfer engine that exposes a live Free Energy stream via Rocket.Subject.
This enables real-time capture of Bethe Free Energy values as they are computed during inference.
"""
mutable struct EngineWithFEStream
    engine::Any  # The underlying RxInfer engine
    free_energy::Rocket.Subject{Float64}  # Live FE stream
    _fe_subscription::Union{Nothing, Any}  # Internal subscription to engine's FE
    _is_started::Bool
end

"""
    create_engine_with_fe_stream(; kwargs...)

Creates an RxInfer engine with an exposed live Free Energy stream.
All kwargs are passed to RxInfer.infer(), with free_energy=true enforced.
"""
function create_engine_with_fe_stream(; kwargs...)
    # Ensure free_energy is enabled
    kwargs_dict = Dict(kwargs)
    kwargs_dict[:free_energy] = true
    kwargs_dict[:autostart] = false  # We'll start manually to set up subscriptions
    
    # Create the underlying RxInfer engine
    engine = RxInfer.infer(; kwargs_dict...)
    
    # Create the FE subject for live streaming
    fe_subject = Rocket.Subject(Float64)
    
    # Create wrapper
    wrapper = EngineWithFEStream(engine, fe_subject, nothing, false)
    
    # Try to hook into the engine's internal FE computation
    _setup_fe_monitoring!(wrapper)
    
    return wrapper
end

"""
    _setup_fe_monitoring!(wrapper::EngineWithFEStream)

Internal function to set up monitoring of the engine's Free Energy computation.
This tries to hook into RxInfer's internal FE calculation and emit values to our stream.
"""
function _setup_fe_monitoring!(wrapper::EngineWithFEStream)
    engine = wrapper.engine
    
    # Debug: check what properties the engine has (uncomment for debugging)
    # @info "Engine properties:" props=propertynames(engine)
    
    # Try to access the engine's internal free energy computation
    # RxInfer engines have fe_source, fe_actor, etc. for FE computation
    try
        if hasproperty(engine, :fe_source) && engine.fe_source !== nothing
            # @info "Setting up FE source monitoring"
            # Subscribe to the FE source directly
            wrapper._fe_subscription = subscribe!(engine.fe_source, 
                fe -> begin
                    try
                        Rocket.next!(wrapper.free_energy, Float64(fe))
                    catch e
                        @warn "Failed to emit FE value from fe_source" fe=fe error=e
                    end
                end)
        elseif hasproperty(engine, :free_energy_history)
            # @info "Setting up FE history monitoring"
            # Monitor the free energy history for new entries
            wrapper._fe_subscription = _monitor_fe_history(wrapper)
        elseif hasproperty(engine, :free_energy)
            # @info "Setting up direct FE stream bridging"
            # If engine already exposes a stream, bridge it
            wrapper._fe_subscription = subscribe!(engine.free_energy, 
                fe -> Rocket.next!(wrapper.free_energy, Float64(fe)))
        else
            @warn "No FE monitoring method found - engine has no suitable FE property"
        end
    catch e
        @warn "Failed to set up FE monitoring" error=e
    end
end

"""
    _monitor_fe_history(wrapper::EngineWithFEStream)

Monitors the engine's free_energy_history for new entries and emits them to our stream.
This is a polling-based approach that checks for new FE values periodically.
"""
function _monitor_fe_history(wrapper::EngineWithFEStream)
    engine = wrapper.engine
    last_fe_count = Ref(0)
    
    # Create a timer-based monitoring task
    timer_task = @async begin
        while wrapper._is_started
            try
                if hasproperty(engine, :free_energy_history) && engine.free_energy_history !== nothing
                    fe_history = engine.free_energy_history
                    current_count = length(fe_history)
                    
                    # Emit any new FE values
                    if current_count > last_fe_count[]
                        for i in (last_fe_count[] + 1):current_count
                            if i <= length(fe_history)
                                fe_val = fe_history[i]
                                try
                                    Rocket.next!(wrapper.free_energy, Float64(fe_val))
                                catch e
                                    @warn "Failed to emit FE value" fe_val=fe_val error=e
                                end
                            end
                        end
                        last_fe_count[] = current_count
                    end
                end
            catch e
                @warn "Error in FE monitoring" error=e
            end
            sleep(0.005)  # Check every 5ms for more responsive capture
        end
    end
    
    return timer_task
end

"""
    RxInfer.start(wrapper::EngineWithFEStream)

Start the wrapped engine and begin FE monitoring.
"""
function RxInfer.start(wrapper::EngineWithFEStream)
    wrapper._is_started = true
    
    # Start the underlying engine
    RxInfer.start(wrapper.engine)
    
    return wrapper
end

"""
    stop!(wrapper::EngineWithFEStream)

Stop the engine and clean up resources.
"""
function stop!(wrapper::EngineWithFEStream)
    wrapper._is_started = false
    
    # Clean up subscription
    if wrapper._fe_subscription !== nothing
        try
            # If it's a task, wait for it to finish
            if wrapper._fe_subscription isa Task
                wait(wrapper._fe_subscription)
            end
        catch
        end
        wrapper._fe_subscription = nothing
    end
    
    # Complete the FE stream
    try
        Rocket.complete!(wrapper.free_energy)
    catch
    end
    
    return wrapper
end

# Forward property access to the underlying engine
function Base.getproperty(wrapper::EngineWithFEStream, name::Symbol)
    if name in (:engine, :free_energy, :_fe_subscription, :_is_started)
        return getfield(wrapper, name)
    else
        # Forward to the underlying engine
        return getproperty(wrapper.engine, name)
    end
end

function Base.hasproperty(wrapper::EngineWithFEStream, name::Symbol)
    if name in (:engine, :free_energy, :_fe_subscription, :_is_started)
        return true
    else
        return hasproperty(wrapper.engine, name)
    end
end

end # module
