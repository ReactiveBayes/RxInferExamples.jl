#!/usr/bin/env julia

# Simple entry point script to run drone examples
# with various configurations

# Run only 2D example with smaller horizon
function run_2d_quick()
    println("🛩️ Running 2D drone example with quick settings...")
    run(`julia "Drone Dynamics.jl" --no-3d --horizon-2d 10 --dt 0.1`)
end

# Run only 3D example with smaller horizon
function run_3d_quick()
    println("🚁 Running 3D drone example with quick settings...")
    run(`julia "Drone Dynamics.jl" --no-2d --horizon-3d 5 --dt 0.1`)
end

# Run both examples with reduced parameters for quicker execution
function run_both_quick()
    println("🔄 Running both 2D and 3D examples with quick settings...")
    run(`julia "Drone Dynamics.jl" --horizon-2d 10 --horizon-3d 5 --dt 0.1`)
end

# Run with full quality for visualization
function run_full_quality()
    println("🌟 Running both examples with full quality settings...")
    run(`julia "Drone Dynamics.jl" --horizon-2d 15 --horizon-3d 8 --dt 0.1 --fps 30`)
end

# Interactive mode to guide users through examples
function run_interactive()
    println("""
    🚀 Welcome to the Interactive Drone Dynamics Demo! 🚀
    
    This interactive mode will guide you through the examples step by step.
    Let's start by selecting which example to run:
    
    1) 2D Drone example only (fastest)
    2) 3D Drone example only
    3) Both examples with reduced parameters
    4) Both examples with full quality settings
    5) Custom configuration
    
    Please enter your choice [1-5]:
    """)
    
    choice = readline()
    
    if choice == "1"
        run_2d_quick()
    elseif choice == "2"
        run_3d_quick()
    elseif choice == "3"
        run_both_quick()
    elseif choice == "4"
        run_full_quality()
    elseif choice == "5"
        # Custom configuration
        println("\n📋 Custom Configuration Setup")
        
        # Get which examples to run
        println("\nWhich examples would you like to run?")
        println("1) Only 2D")
        println("2) Only 3D")
        println("3) Both 2D and 3D")
        example_choice = readline()
        
        run_2d = example_choice == "1" || example_choice == "3"
        run_3d = example_choice == "2" || example_choice == "3"
        
        # Get time horizon
        horizon_2d = 15
        horizon_3d = 8
        dt = 0.1
        fps = 20
        
        if run_2d
            print("\n2D Horizon (default: 15, lower is faster): ")
            input = readline()
            if !isempty(input)
                horizon_2d = parse(Int, input)
            end
        end
        
        if run_3d
            print("\n3D Horizon (default: 8, lower is faster): ")
            input = readline()
            if !isempty(input)
                horizon_3d = parse(Int, input)
            end
        end
        
        print("\nTime step (default: 0.1, higher is faster but less accurate): ")
        input = readline()
        if !isempty(input)
            dt = parse(Float64, input)
        end
        
        print("\nAnimation FPS (default: 20): ")
        input = readline()
        if !isempty(input)
            fps = parse(Int, input)
        end
        
        # Construct command
        cmd = "julia \"Drone Dynamics.jl\""
        if !run_2d
            cmd *= " --no-2d"
        end
        if !run_3d
            cmd *= " --no-3d"
        end
        cmd *= " --horizon-2d $horizon_2d --horizon-3d $horizon_3d --dt $dt --fps $fps"
        
        println("\n🚀 Running with custom configuration:")
        println(cmd)
        run(`bash -c $cmd`)
    else
        println("Invalid choice. Please run again and select a number between 1 and 5.")
        return
    end
end

# Print a fancy banner
function print_banner()
    println("""
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║  🚁 🛩️  RxInfer.jl Drone Dynamics Demo  🛩️ 🚁  ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """)
end

# Get command line arguments
if length(ARGS) == 0
    print_banner()
    println("""
    Usage: julia run_examples.jl [option]
      
    Options:
      2d-quick     - Run only 2D example with reduced parameters
      3d-quick     - Run only 3D example with reduced parameters
      both-quick   - Run both examples with reduced parameters
      full         - Run both examples with full quality settings
      interactive  - Interactive mode with guided configuration
      
    Running interactive mode by default.
    """)
    run_interactive()
else
    option = ARGS[1]
    print_banner()
    if option == "2d-quick"
        run_2d_quick()
    elseif option == "3d-quick"
        run_3d_quick()
    elseif option == "both-quick"
        run_both_quick()
    elseif option == "full"
        run_full_quality()
    elseif option == "interactive"
        run_interactive()
    else
        println("Unknown option: $option")
        println("Try: 2d-quick, 3d-quick, both-quick, full, or interactive")
    end
end 