# Diagnostics Module Tests
# Comprehensive tests for diagnostics.jl functionality

using Test
using LinearAlgebra
using Statistics

# Load modules if not already loaded
if !isdefined(Main, :Config)
    include("../config.jl")
end
if !isdefined(Main, :Diagnostics)
    include("../src/diagnostics.jl")
end
using .Config
using .Diagnostics

@testset "Memory Tracer" begin
    @testset "Memory Tracer Creation" begin
        tracer = MemoryTracer()
        @test tracer isa MemoryTracer
        @test tracer.enabled == true
        @test length(tracer.memory_mb) == 0
    end
    
    @testset "Memory Tracing" begin
        tracer = MemoryTracer()
        trace_memory!(tracer)
        
        @test length(tracer.memory_mb) > 0
        @test tracer.memory_mb[1] > 0
    end
    
    @testset "Multiple Memory Traces" begin
        tracer = MemoryTracer()
        for _ in 1:10
            trace_memory!(tracer)
        end
        
        @test length(tracer.memory_mb) == 10
        @test all(tracer.memory_mb .> 0)
    end
    
    @testset "Memory Summary" begin
        tracer = MemoryTracer()
        for _ in 1:10
            trace_memory!(tracer)
        end
        
        summary = get_memory_summary(tracer)
        @test haskey(summary, "peak_memory_mb")
        @test haskey(summary, "avg_memory_mb")
        @test haskey(summary, "measurements")
        @test summary["measurements"] == 10
        @test summary["peak_memory_mb"] >= summary["avg_memory_mb"]
    end
    
    @testset "Disabled Tracer" begin
        tracer = MemoryTracer(enabled=false)
        trace_memory!(tracer)
        @test length(tracer.memory_mb) == 0
    end
end

@testset "Performance Profiler" begin
    @testset "Profiler Creation" begin
        profiler = PerformanceProfiler()
        @test profiler isa PerformanceProfiler
        @test profiler.enabled == true
        @test length(profiler.operation_times) == 0
    end
    
    @testset "Timer Operations" begin
        profiler = PerformanceProfiler()
        
        start_timer!(profiler, "test_op")
        sleep(0.01)
        stop_timer!(profiler, "test_op")
        
        @test haskey(profiler.operation_times, "test_op")
        @test length(profiler.operation_times["test_op"]) == 1
        @test profiler.operation_times["test_op"][1] >= 0.01
    end
    
    @testset "Multiple Operations" begin
        profiler = PerformanceProfiler()
        
        for i in 1:5
            start_timer!(profiler, "op_$i")
            sleep(0.001)
            stop_timer!(profiler, "op_$i")
        end
        
        @test length(profiler.operation_times) == 5
    end
    
    @testset "Repeated Operations" begin
        profiler = PerformanceProfiler()
        
        for _ in 1:10
            start_timer!(profiler, "repeated_op")
            sleep(0.001)
            stop_timer!(profiler, "repeated_op")
        end
        
        @test length(profiler.operation_times["repeated_op"]) == 10
    end
    
    @testset "Performance Summary" begin
        profiler = PerformanceProfiler()
        
        for _ in 1:10
            start_timer!(profiler, "test")
            sleep(0.001)
            stop_timer!(profiler, "test")
        end
        
        summary = get_performance_summary(profiler)
        @test haskey(summary, "operations")
        @test haskey(summary["operations"], "test")
        
        op_stats = summary["operations"]["test"]
        @test haskey(op_stats, "count")
        @test haskey(op_stats, "total_time")
        @test haskey(op_stats, "avg_time")
        @test haskey(op_stats, "min_time")
        @test haskey(op_stats, "max_time")
        
        @test op_stats["count"] == 10
        @test op_stats["total_time"] >= 0.01
    end
end

@testset "Belief Tracker" begin
    @testset "Belief Tracker Creation" begin
        tracker = BeliefTracker()
        @test tracker isa BeliefTracker
        @test tracker.enabled == true
        @test length(tracker.belief_means) == 0
    end
    
    @testset "Recording Beliefs" begin
        tracker = BeliefTracker()
        record_belief!(tracker, 1, [1.0, 2.0], Matrix{Float64}(I, 2, 2))
        
        @test length(tracker.belief_means) == 1
        @test length(tracker.belief_covs) == 1
        @test tracker.belief_means[1] == [1.0, 2.0]
    end
    
    @testset "Multiple Belief Records" begin
        tracker = BeliefTracker()
        
        for t in 1:10
            record_belief!(tracker, t, [Float64(t), Float64(t+1)], Matrix{Float64}(I, 2, 2))
        end
        
        @test length(tracker.belief_means) == 10
        @test length(tracker.belief_changes) == 9
    end
    
    @testset "Belief Changes Calculation" begin
        tracker = BeliefTracker()
        
        record_belief!(tracker, 1, [1.0, 2.0], Matrix{Float64}(I, 2, 2))
        record_belief!(tracker, 2, [1.5, 2.5], Matrix{Float64}(I, 2, 2))
        
        @test length(tracker.belief_changes) == 1
        expected_change = sqrt((0.5)^2 + (0.5)^2)
        @test tracker.belief_changes[1] ≈ expected_change atol=1e-10
    end
    
    @testset "Belief Summary" begin
        tracker = BeliefTracker()
        
        for t in 1:10
            record_belief!(tracker, t, [Float64(t), Float64(t+1)], Matrix{Float64}(I, 2, 2))
        end
        
        summary = get_belief_summary(tracker)
        @test haskey(summary, "measurements")
        @test haskey(summary, "avg_belief_change")
        @test haskey(summary, "avg_uncertainty")
        @test haskey(summary, "final_uncertainty")
        @test summary["measurements"] == 10
    end
end

@testset "Prediction Tracker" begin
    @testset "Prediction Tracker Creation" begin
        tracker = PredictionTracker()
        @test tracker isa PredictionTracker
        @test tracker.enabled == true
        @test length(tracker.predictions) == 0
    end
    
    @testset "Recording Predictions" begin
        tracker = PredictionTracker()
        preds = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]
        actual = [1.05, 2.05]
        
        record_predictions!(tracker, preds, actual)
        
        @test length(tracker.predictions) == 1
        @test length(tracker.actual_states) == 1
        # prediction_errors is computed after second prediction
    end
    
    @testset "Prediction Error Calculation" begin
        tracker = PredictionTracker()
        # Need two predictions to compute error
        preds1 = [[1.0, 2.0], [1.1, 2.1]]
        actual1 = [1.0, 2.0]
        record_predictions!(tracker, preds1, actual1)
        
        preds2 = [[1.05, 2.05], [1.15, 2.15]]
        actual2 = [1.05, 2.05]
        record_predictions!(tracker, preds2, actual2)
        
        # Now we should have errors
        @test length(tracker.prediction_errors) >= 1
    end
    
    @testset "Prediction Summary" begin
        tracker = PredictionTracker()
        
        for _ in 1:10
            preds = [[1.0, 2.0], [1.1, 2.1]]
            actual = [1.05, 2.05]
            record_predictions!(tracker, preds, actual)
        end
        
        summary = get_prediction_summary(tracker)
        @test haskey(summary, "measurements")
        @test haskey(summary, "avg_error_by_horizon")
        @test summary["measurements"] == 10
    end
end

@testset "Free Energy Tracker" begin
    @testset "Free Energy Tracker Creation" begin
        tracker = FreeEnergyTracker()
        @test tracker isa FreeEnergyTracker
        @test tracker.enabled == true
        @test length(tracker.free_energies) == 0
    end
    
    @testset "Recording Free Energy" begin
        tracker = FreeEnergyTracker()
        record_free_energy!(tracker, 1, 100.0)
        
        @test length(tracker.free_energies) == 1
        @test tracker.free_energies[1] == 100.0
    end
    
    @testset "Free Energy Reduction" begin
        tracker = FreeEnergyTracker()
        
        for t in 1:10
            record_free_energy!(tracker, t, 100.0 - t)
        end
        
        summary = get_free_energy_summary(tracker)
        @test haskey(summary, "fe_reduction")
        @test summary["fe_reduction"] > 0
    end
    
    @testset "Free Energy Summary" begin
        tracker = FreeEnergyTracker()
        
        for t in 1:10
            record_free_energy!(tracker, t, 100.0 - Float64(t))
        end
        
        summary = get_free_energy_summary(tracker)
        @test haskey(summary, "measurements")
        @test haskey(summary, "final_free_energy")
        @test haskey(summary, "min_free_energy")
        @test haskey(summary, "max_free_energy")
        @test haskey(summary, "fe_reduction")
        @test summary["measurements"] == 10
        @test summary["fe_reduction"] > 0
    end
end

@testset "Diagnostics Collector" begin
    @testset "Collector Creation" begin
        collector = DiagnosticsCollector()
        @test collector isa DiagnosticsCollector
        @test collector.memory_tracer isa MemoryTracer
        @test collector.performance_profiler isa PerformanceProfiler
        @test collector.belief_tracker isa BeliefTracker
        @test collector.prediction_tracker isa PredictionTracker
        @test collector.free_energy_tracker isa FreeEnergyTracker
    end
    
    @testset "Comprehensive Summary" begin
        collector = DiagnosticsCollector()
        
        # Add some data
        trace_memory!(collector.memory_tracer)
        record_belief!(collector.belief_tracker, 1, [1.0, 2.0], Matrix{Float64}(I, 2, 2))
        record_free_energy!(collector.free_energy_tracker, 1, 100.0)
        
        summary = get_comprehensive_summary(collector)
        @test haskey(summary, "memory")
        @test haskey(summary, "performance")
        @test haskey(summary, "beliefs")
        @test haskey(summary, "predictions")
        @test haskey(summary, "free_energy")
    end
    
    @testset "Print Diagnostics Report" begin
        collector = DiagnosticsCollector()
        
        # Add data
        for t in 1:5
            record_belief!(collector.belief_tracker, t, [Float64(t), Float64(t+1)], Matrix{Float64}(I, 2, 2))
            record_free_energy!(collector.free_energy_tracker, t, 100.0 - Float64(t))
        end
        
        # Should not error
        @test_nowarn print_diagnostics_report(collector)
    end
end

@testset "Diagnostics Performance" begin
    @testset "Memory Tracing Overhead" begin
        tracer = MemoryTracer()
        
        elapsed = @elapsed begin
            for _ in 1:100
                trace_memory!(tracer)
            end
        end
        
        # Should be fast
        @test elapsed < 0.5
    end
    
    @testset "Profiler Overhead" begin
        profiler = PerformanceProfiler()
        
        elapsed = @elapsed begin
            for _ in 1:1000
                start_timer!(profiler, "test")
                stop_timer!(profiler, "test")
            end
        end
        
        # Should be very fast
        @test elapsed < 0.1
    end
end

println("✅ Diagnostics tests completed successfully")

