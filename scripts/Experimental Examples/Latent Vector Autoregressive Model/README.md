# Latent Vector Autoregressive Model (LVAR)

This directory contains an implementation of a Latent Vector Autoregressive Model using RxInfer.jl. The model is designed to learn from and predict multivariate time series data with underlying autoregressive processes.

## Overview

The Latent Vector Autoregressive Model combines multiple autoregressive (AR) processes into a single unified model. It can be used for:

- Multivariate time series forecasting
- Learning relationships between multiple observed signals
- Capturing the latent structure of complex dynamical systems

```mermaid
graph TD
    A[Input Data] --> B[LVAR Model]
    B --> C[Inference]
    C --> D[Predictions]
    D --> E[Visualization]
    D --> F[Analysis]
    E --> G[Output Results]
    F --> G
```

## Files

- `LVAR_Refactored.jl` - Main implementation with modular design
- `lva_visualization.jl` - Module for visualizations and data export
- `TECHNICAL_README.md` - Formal model description, derivations, and theory
- `results/` - Directory containing output results (created during execution)

## Complete Workflow

```mermaid
flowchart TD
    A[Data Generation] --> B[Model Definition]
    B --> C[Inference]
    C --> D[Prediction]
    D --> E[Analysis & Visualization]
    E --> F[Export Results]
    
    subgraph "Data Generation"
    A1[Generate AR Parameters] --> A2[Generate AR Processes]
    A2 --> A3[Create Observations]
    A3 --> A4[Split Training/Test]
    end
    
    subgraph "Model Definition"
    B1[Form Priors] --> B2[Define AR Sequences]
    B2 --> B3[Define Linear Combinations]
    B3 --> B4[Link to Observations]
    end
    
    subgraph "Inference"
    C1[Initialize Parameters] --> C2[Apply Constraints]
    C2 --> C3[Run Message Passing]
    C3 --> C4[Extract Posteriors]
    C4 --> C5[Track Parameter History]
    end
    
    subgraph "Analysis & Visualization"
    E1[Calculate Metrics] --> E2[Generate Basic Plots]
    E2 --> E3[Generate Advanced Visualizations]
    E3 --> E4[Create Animation]
    E4 --> E5[Visualize Parameter Evolution]
    end
    
    A --> A1
    B --> B1
    C --> C1
    E --> E1
```

## Model Architecture

The LVAR model consists of:

1. **Multiple AR processes**: Each with its own order and parameters
2. **Latent space representation**: Connecting the AR processes to observations
3. **Variational inference**: For learning parameters and making predictions

```mermaid
graph LR
    A[Input Data] --> B[Autoregressive Processes]
    B --> C[Linear Combination]
    C --> D[Observations]
    E[Prior Distributions] --> B
    F[Noise Model] --> D
    
    subgraph "Latent AR Processes"
    P1[Process 1]
    P2[Process 2]
    P3[Process 3]
    Pn[Process n]
    end
    
    subgraph "Model Parameters"
    T1[Precision γ]
    T2[Coefficients θ]
    T3[Output Weights]
    end
    
    T1 --> P1
    T1 --> P2
    T1 --> P3
    T1 --> Pn
    
    T2 --> P1
    T2 --> P2
    T2 --> P3
    T2 --> Pn
    
    P1 --> C
    P2 --> C
    P3 --> C
    Pn --> C
    
    T3 --> C
```

## How to Run

Simply execute the main script:

```bash
julia LVAR_Refactored.jl
```

## Model Parameters

The default configuration uses:
- 20 AR processes, each with order 5
- 120 time steps of data (100 for training, 20 for testing)

You can adjust these parameters in the `main()` function.

```mermaid
classDiagram
    class ModelParameters {
        orders: Array[Int]
        n_samples: Int
        n_missing: Int
        n_ar_processes: Int
    }
    
    class ARProcess {
        order: Int
        coefficients θ: Array[Float64]
        precision γ: Float64
        states x: Array[Float64]
    }
    
    ModelParameters "1" --> "*" ARProcess: configures
```

## Inference Process

The model uses variational message passing through RxInfer.jl to perform Bayesian inference on the model parameters and latent states.

```mermaid
flowchart LR
    A[Initialize Posteriors] --> B[Form Mean Field Constraints]
    B --> C[Run Message Passing]
    C --> D[Update Posteriors]
    D --> E{Converged?}
    E -->|No| C
    E -->|Yes| F[Extract Results]
    
    subgraph "Parameter Tracking"
    P1[Track θ]
    P2[Track γ]
    P3[Track τ]
    end
    
    C --> P1
    C --> P2
    C --> P3
```

The implementation now tracks the evolution of key model parameters (θ, γ, τ) across iterations, allowing for convergence analysis and better understanding of the inference process.

## Outputs

The script generates various outputs in a timestamped directory under `results/`:

### Visualizations

```mermaid
classDiagram
    class Visualizations {
        lvar_predictions.png
        error_heatmap.png
        correlation_plots.png
        uncertainty_analysis.png
        parameter_evolution.png
        complexity_analysis.png
        residual_analysis.png
        prediction_animation.gif
    }
```

- **Basic predictions**: Time series plots showing true values, observations, and predictions
- **Error heatmap**: Visualization of prediction errors across processes and time
- **Correlation plots**: Analysis of correlations between processes
- **Uncertainty visualization**: Analysis of prediction uncertainties
- **Parameter evolution**: Plot showing convergence of AR coefficients (θ), process precisions (γ), and observation precision (τ) during inference
- **Model complexity analysis**: SVD-based analysis of the latent structure
- **Residual analysis**: Statistical analysis of prediction residuals
- **Prediction animation**: GIF animation showing predictions evolving over time

### Data Files

```mermaid
classDiagram
    class DataFiles {
        lvar_predictions.csv: Contains time indices, true values, predicted means, and prediction std devs
        model_statistics.json: Contains overall statistics, process statistics, and model parameters
    }
```

The model_statistics.json contains:
- **overall_statistics**: avg_rmse, median_rmse, min_rmse, max_rmse, rmse_std, avg_uncertainty
- **process_statistics**: rmse_by_process (RMSE for each individual process)
- **model_parameters**: n_processes, n_samples, test_samples, training_samples

## Analysis Example

From a sample output in the `results/2025-04-29_082115` directory:

```mermaid
graph TD
    A[Overall Performance] --> B[Average RMSE: 1.999]
    A --> C[Min RMSE: 0.742]
    A --> D[Max RMSE: 3.299]
    A --> E[Average Uncertainty: 1.205]
```

Model configuration for this example:
- 20 AR processes of order 5
- 120 samples (100 for training, 20 for testing)
- Process-specific RMSE varies from 0.742 to 3.299

## Performance Metrics

The script reports various metrics including:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Execution time and inference speed

```mermaid
graph LR
    A[Performance Evaluation] --> B[Accuracy Metrics]
    A --> C[Efficiency Metrics]
    B --> D[RMSE]
    B --> E[MAE]
    B --> F[MAPE]
    C --> G[Execution Time]
    C --> H[Inference Speed]
```

## Dependencies

Required Julia packages:
- RxInfer
- Random, LinearAlgebra, Dates, Printf, Statistics
- Plots, DelimitedFiles, StatsBase, JSON
- Distributions, SpecialFunctions

## Extending the Model

To adapt the model for your own data:
1. Modify the `generate_data()` function to load your data instead of synthetic data
2. Adjust the model parameters (orders, number of processes) to match your problem
3. Customize the visualizations as needed

```mermaid
flowchart TD
    A[Your Data] --> B[Data Preparation]
    B --> C[LVAR Model]
    C --> D[Custom Visualization]
    C --> E[Performance Analysis]
    E --> F[Model Optimization]
    F -->|Adjust Parameters| C
```

## Contributing

Feel free to improve the implementation or add new features. When making changes, please:
1. Add appropriate error handling
2. Include informative logging
3. Update visualizations as needed
4. Document any new functionality 