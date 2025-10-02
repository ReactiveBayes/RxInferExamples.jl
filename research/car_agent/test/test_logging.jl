# Logging Module Tests
# Comprehensive tests for logging.jl functionality

using Test

# Load modules if not already loaded
if !isdefined(Main, :Config)
    include("../config.jl")
end
if !isdefined(Main, :LoggingUtils)
    include("../src/logging.jl")
end
using .Config
using .LoggingUtils

@testset "Progress Bar" begin
    @testset "Progress Bar Creation" begin
        pb = ProgressBar(10)
        @test pb isa ProgressBar
        @test pb.total == 10
        @test pb.current == 0
    end
    
    @testset "Progress Bar Update" begin
        pb = ProgressBar(10)
        update!(pb, 5)
        @test pb.current == 5
    end
    
    @testset "Progress Bar Completion" begin
        pb = ProgressBar(10)
        for i in 1:10
            update!(pb, i)
        end
        finish!(pb)
        @test pb.current == 10
    end
    
    @testset "Progress Bar Edge Cases" begin
        pb = ProgressBar(1)
        update!(pb, 1)
        finish!(pb)
        @test pb.current == 1
    end
end

@testset "Logging Operations" begin
    @testset "Log Event Creation" begin
        # Should log the event (logs to stderr by default)
        @test_logs (:info, "test_event") log_event("test_event", Dict{String, Any}("key" => "value"))
    end
    
    @testset "Log Agent Step" begin
        # Should log agent step
        @test_logs (:info, "agent_step") log_agent_step(1, [1.0], [2.0]; free_energy=100.0)
    end
end

@testset "Logging Performance" begin
    @testset "Log Event Performance" begin
        elapsed = @elapsed begin
            for i in 1:1000
                log_event("test", Dict{String, Any}("iteration" => i))
            end
        end
        
        # Should be reasonably fast
        @test elapsed < 1.0
    end
    
    @testset "Progress Bar Performance" begin
        elapsed = @elapsed begin
            pb = ProgressBar(1000)
            for i in 1:1000
                update!(pb, i)
            end
            finish!(pb)
        end
        
        # Should be very fast
        @test elapsed < 0.5
    end
end

@testset "Logging Edge Cases" begin
    @testset "Empty Event Data" begin
        @test_logs (:info, "empty") log_event("empty", Dict{String, Any}())
    end
    
    @testset "Large Event Data" begin
        large_data = Dict{String, Any}("data" => randn(1000))
        @test_logs (:info, "large") log_event("large", large_data)
    end
    
    @testset "Special Characters in Event Names" begin
        @test_logs (:info, "test/event:name") log_event("test/event:name", Dict{String, Any}("key" => "value"))
    end
end

println("✅ Logging tests completed successfully")

