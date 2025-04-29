# Financial Latent Vector Autoregressive Model (F-LVAR)

This directory contains an implementation of a Latent Vector Autoregressive Model for financial time series analysis using RxInfer.jl. The model is designed for denoising and trend analysis of financial data.

Perfect for when you're cooking up some financial insights with F-LVAR, like:
- Rolling Pin-dicators 📊🍳 (Technical Indicators)
- Whisk Management 🥣💼 (Risk Management)
- Taco'ing about trends 🌮📈 (Trend Analysis)
- Trade du Jour 🍲💹 (Daily Trading Strategy)
- Getting that bread 🍞💰 (Profit Generation)
- Financial flavors and mouthfeels 😋📉📈 (Market Sentiment)
 b- Dough Jones Industrial Average 💰🍞 (Market Indices)
- Market Cap-puccino ☕📊 (Market Capitalization)
- Volatility Soufflé 📈📉🍮 (Handling Market Volatility)
- Hedging your bets with herbs 🌿🛡️ (Hedging Strategies)
- Latent Vector Apple Returns 🍎📉📈 (Analyzing Asset Returns with LVAR)
- F-LVAR: Financial Latte Vector Autoregressive Roast ☕️🔥 (Our Model!)
- And whipping up more financial feasts! 👨‍🍳👩‍🍳

## Overview

The Financial Latent Vector Autoregressive Model (F-LVAR) applies the LVAR framework to financial time series data, providing:

- Noise reduction for volatile financial signals
- Trend identification and forecasting
- Multivariate analysis of correlated financial instruments
- Uncertainty quantification for risk assessment

## Files

- `FLVAR.jl` - Main implementation for financial time series
- `flva_visualization.jl` - Enhanced visualization module for financial data
- `data_sources.jl` - Module for synthetic and API-based financial data
- `results/` - Directory containing output results (created during execution)

## Features

1. **Financial Data Sources**:
   - Synthetic financial time series generation
   - API-based data fetching for real financial instruments
   - Support for multiple data formats and frequencies

2. **Financial-Specific Modeling**:
   - Volatility clustering detection
   - Mean-reversion and momentum modeling
   - Seasonal and cyclical component extraction

3. **Enhanced Visualization**:
   - Candlestick and OHLC charts
   - Technical indicators overlay
   - Risk metrics visualization
   - Performance analytics

## Data Generation

The model can work with both synthetic financial data and real market data:

1. **Synthetic Data**: Generated with realistic financial properties like:
   - Random walks with drift
   - GARCH-like volatility patterns
   - Jump diffusion processes
   - Correlated asset behavior

2. **Real Data**: Fetched via API from sources like:
   - Alpha Vantage
   - Yahoo Finance
   - Custom data sources

## Running the Model

```bash
julia FLVAR.jl
```

By default, it uses synthetic data. To use real financial data:

```bash
julia FLVAR.jl --data-source=api --symbols=AAPL,MSFT,GOOGL --start-date=2020-01-01
```

## Performance Metrics

The model reports financial-specific metrics including:
- Directional accuracy
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown
- Standard technical indicators (RSI, MACD, etc.)

## Example Applications

1. **Noise Filtering**: Clean noisy financial signals to identify underlying trends
2. **Multi-Asset Analysis**: Model correlations between related financial instruments
3. **Risk Forecasting**: Predict volatility patterns and uncertainty in asset movements
4. **Anomaly Detection**: Identify unusual patterns or regime changes in financial data 