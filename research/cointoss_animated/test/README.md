# Coin Toss Model - Test Suite Documentation

## Overview

Comprehensive test suite for the Coin Toss Model research fork with modular organization, extensive logging, and performance benchmarking.

## Test Structure

### Main Test Runner
- **`runtests.jl`** - Main test orchestrator that runs all test modules with comprehensive logging

### Modular Test Files

1. **`test_model.jl`** - CoinTossModel module tests
   - Data generation (basic, edge cases, reproducibility, invalid inputs)
   - RxInfer model definition
   - Analytical posterior computation
   - Posterior statistics (basic, edge cases, various credible levels)
   - Log marginal likelihood (basic, properties, edge cases)
   - Conjugate property verification
   - Statistical consistency checks

2. **`test_inference.jl`** - CoinTossInference module tests
   - InferenceResult structure validation
   - Basic inference execution
   - Convergence detection and diagnostics
   - KL divergence computation (basic, various distributions)
   - Expected log likelihood (basic, edge cases)
   - Posterior predictive checks (basic, reproducibility, edge cases)
   - Free energy tracking
   - Analytical vs numerical agreement
   - Information gain analysis
   - Variance reduction analysis

3. **`test_visualization.jl`** - CoinTossVisualization module tests
   - Theme colors (default, dark, colorblind)
   - Prior-posterior plots (basic, various configurations)
   - Convergence plots (basic, various cases)
   - Data histograms (basic, edge cases)
   - Credible interval plots (basic, various levels)
   - Predictive plots (basic, various scenarios)
   - Comprehensive dashboards (with/without free energy, various themes)
   - Animation creation (basic, various configurations)
   - Plot saving functionality
   - Integration workflows
   - Consistency across themes

4. **`test_utils.jl`** - CoinTossUtils module tests
   - Logging setup
   - Timer functionality (basic, elapsed time, multiple timers)
   - ProgressBar (basic, various totals)
   - CSV export (basic, nested dictionaries)
   - JSON export (basic, nested structures)
   - Dictionary flattening (basic, deep nesting, arrays)
   - Experiment result saving
   - Directory creation utilities
   - Dictionary logging
   - Time formatting (various durations)
   - Byte formatting (various sizes)
   - Summary statistics (basic, edge cases)
   - Bernoulli confidence intervals (basic, various levels, edge cases)
   - Utility integration

5. **`test_performance.jl`** - Performance and benchmark tests
   - Data generation performance (small, medium, large datasets)
   - Analytical posterior performance
   - RxInfer inference performance
   - KL divergence performance
   - Posterior statistics performance
   - Visualization performance
   - Dashboard creation performance
   - Export performance
   - Dictionary flattening performance
   - End-to-end workflow performance
   - Memory efficiency tests
   - Scalability tests
   - Convergence speed analysis
   - Parallel execution readiness

## Running Tests

### Run All Tests
```bash
cd research/cointoss
julia --project=. test/runtests.jl
```

### Run Specific Test Module
```bash
cd research/cointoss
julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/test_model.jl")'
```

### Run with Package Manager
```bash
cd research/cointoss
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Test Logging

### Log Output Locations
- **Console**: Real-time test progress and results
- **File**: `outputs/logs/test_run_YYYY-MM-DD_HH-MM-SS.log`

### Log Levels
- `Info`: Standard test progress, module loading, test completion
- `Warn`: Configuration issues, non-critical failures
- `Error`: Test failures, critical errors

### What Gets Logged
- Test suite initialization
- Module loading status
- Individual test set execution
- Performance benchmark results
- Test timing information
- Comprehensive summary with:
  - Total test duration
  - Modules tested
  - Test categories
  - Coverage summary
  - Final status

## Test Coverage

### Comprehensive Coverage Includes

#### Functionality Testing
- ✓ All exported functions
- ✓ All public interfaces
- ✓ Internal helper functions (where critical)
- ✓ Configuration validation
- ✓ Error handling
- ✓ Edge cases

#### Scenario Testing
- ✓ Normal use cases
- ✓ Boundary conditions
- ✓ Invalid inputs
- ✓ Large datasets
- ✓ Empty/minimal datasets
- ✓ Extreme parameter values

#### Integration Testing
- ✓ Module interactions
- ✓ End-to-end workflows
- ✓ Analytical vs numerical agreement
- ✓ Data pipeline validation
- ✓ Visualization pipeline

#### Performance Testing
- ✓ Execution speed benchmarks
- ✓ Memory efficiency
- ✓ Scalability analysis
- ✓ Convergence speed
- ✓ Export performance

## Test Metrics

### Expected Performance Benchmarks
- Data generation (n=100): < 0.01s
- Analytical posterior: < 0.001s (closed form)
- RxInfer inference (n=50, iter=10): < 2s
- KL divergence: < 0.0001s per computation
- Visualization (single plot): < 5s
- Dashboard creation: < 15s
- CSV export: < 2s
- JSON export: < 2s
- End-to-end workflow: < 20s

### Test Count Summary
- **Model Tests**: ~15 test sets, ~80+ individual tests
- **Inference Tests**: ~15 test sets, ~90+ individual tests
- **Visualization Tests**: ~15 test sets, ~70+ individual tests
- **Utils Tests**: ~20 test sets, ~100+ individual tests
- **Performance Tests**: ~12 test sets, ~50+ benchmarks
- **Total**: ~75+ test sets, 400+ individual assertions

## Continuous Integration

### CI-Ready Features
- Non-interactive execution
- Comprehensive logging
- Performance benchmarks
- Test isolation
- Parallel-ready structure
- Reproducible results (seeded RNG)

### Recommended CI Configuration
```yaml
test:
  script:
    - julia --project=. -e 'using Pkg; Pkg.instantiate()'
    - julia --project=. test/runtests.jl
  artifacts:
    paths:
      - outputs/logs/
    expire_in: 1 week
```

## Extending Tests

### Adding New Tests to Existing Module

1. Open the appropriate test file (e.g., `test_model.jl`)
2. Add a new `@testset` within the relevant category
3. Include comprehensive logging:
   ```julia
   @testset "New Feature Test" begin
       @info "Testing new feature"
       
       # Your tests here
       @test new_function() == expected_result
       
       @info "New feature tests passed"
   end
   ```

### Adding New Test Module

1. Create `test/test_newmodule.jl`
2. Structure with logging:
   ```julia
   #!/usr/bin/env julia
   using Test
   using Logging
   
   test_logger = ConsoleLogger(stderr, Logging.Info)
   global_logger(test_logger)
   
   @info "Starting NewModule tests"
   
   @testset "NewModule Tests" begin
       # Your test sets here
   end
   
   @info "All NewModule tests completed"
   ```
3. Add to `runtests.jl`:
   ```julia
   @testset "N. NewModule Tests" begin
       @info "="^60
       @info "Running NewModule Tests (test_newmodule.jl)"
       @info "="^60
       include("test_newmodule.jl")
   end
   ```

## Test Best Practices

### Logging
- Log at the start and end of each test set
- Include timing information for performance-critical tests
- Log meaningful context (parameters, data sizes, etc.)
- Use separator lines for visual clarity

### Assertions
- Test both positive and negative cases
- Include edge cases and boundary conditions
- Verify error handling with `@test_throws`
- Use `@test_nowarn` for functions that should not error
- Use appropriate tolerances for floating-point comparisons

### Organization
- Group related tests in `@testset` blocks
- Use descriptive test set names
- Keep tests focused and atomic
- Test one thing per assertion when possible

### Performance
- Use `@elapsed` for timing tests
- Set reasonable performance thresholds
- Test scalability with varying input sizes
- Document expected performance metrics

### Reproducibility
- Always seed RNGs for deterministic tests
- Document any stochastic behavior
- Use consistent test data across related tests
- Clean up temporary files/directories

## Troubleshooting

### Common Issues

1. **Module not found**
   - Ensure you're in the correct directory
   - Check that `Pkg.activate("..")` is working
   - Verify all dependencies are installed

2. **Slow tests**
   - Check performance benchmarks in `test_performance.jl`
   - Reduce iteration counts for debugging
   - Use `showprogress=false` in inference calls

3. **Visualization failures**
   - Ensure Plots.jl backend is configured
   - Check for headless environment issues
   - Verify output directories exist

4. **Random test failures**
   - Check that RNG is properly seeded
   - Verify tolerance values for stochastic tests
   - Look for race conditions in parallel code

### Debug Mode

Run tests with verbose output:
```bash
julia --project=. test/runtests.jl --verbose
```

Run specific test with debugging:
```bash
julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/test_model.jl")' --debug
```

## Test Maintenance

### Regular Tasks
- Update benchmarks when optimizations are made
- Add tests for new features immediately
- Review and update edge case coverage
- Maintain test documentation
- Clean up obsolete tests

### Version Updates
When updating dependencies:
1. Run full test suite
2. Update performance benchmarks if needed
3. Fix any breaking changes
4. Document compatibility issues

## Summary

This comprehensive test suite ensures:
- ✓ Complete functional coverage
- ✓ Robust error handling
- ✓ Performance monitoring
- ✓ Integration validation
- ✓ Extensive logging for debugging
- ✓ CI/CD readiness
- ✓ Maintainability and extensibility

All tests are designed to be:
- **Comprehensive**: Cover all functionality and edge cases
- **Isolated**: Tests don't depend on each other
- **Reproducible**: Seeded RNG ensures consistent results
- **Documented**: Clear logging and documentation
- **Fast**: Complete suite runs in < 2 minutes
- **Maintainable**: Modular structure for easy updates

