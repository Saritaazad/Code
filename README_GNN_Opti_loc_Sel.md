# GNN-Based Optimal Rain Gauge Network Selection

## Overview
Implements a Graph Attention Network (GAT) combined with LSTM for data-driven selection of optimal meteorological monitoring locations. Uses learned attention weights to rank location importance and identify the most informative subset of stations for network optimization.

## Purpose
Addresses the rain gauge network optimization problem by learning spatial-temporal dependencies between monitoring locations through deep learning, then using attention mechanism scores to identify which locations contribute most to predictive accuracy.

## Architecture

```
Input Data → Graph Construction → GAT-LSTM Model → Attention Weights → Location Ranking → Optimal Network
```

### Model Components
- **GATConv**: Graph Attention Network layer (2 heads, 32 channels)
- **LSTM**: Temporal sequence modeling (128 hidden units)
- **Linear**: Output projection to target variable

## Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK_WINDOW` | 12 | Temporal sequence length for LSTM |
| `N_NEIGHBORS` | 5 | KNN graph connectivity |
| `N_LOCATIONS_TO_SELECT` | 10 | Final network size |
| `EPOCHS` | 20 | Training iterations |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |

## Features Used
- `T2M`, `T2M_MAX`, `T2M_MIN` – 2-meter temperature (mean, max, min)
- `QV2M` – 2-meter specific humidity
- `WD10M`, `WS10M` – 10-meter wind direction and speed
- `Pres` – Surface pressure (target variable)
- `elevation` – Terrain elevation

## Requirements
```
pandas
numpy
torch
torch_geometric
scikit-learn
```

## Input
`data10000.csv` – Gridded meteorological data with columns:
```
longitude, latitude, Year, Month, Day, T2M, T2M_MAX, T2M_MIN, QV2M, WD10M, WS10M, Pres, date_col, elevation
```

## Output
| File | Description |
|------|-------------|
| `GNN_optimal_network_data.csv` | Time series data for selected optimal locations |

## Workflow
1. **Data Loading**: Robust CSV parsing with header detection
2. **Date Construction**: Validates and constructs datetime from Y/M/D columns
3. **Feature Scaling**: StandardScaler normalization
4. **Graph Construction**: KNN with haversine distance metric
5. **Model Training**: GAT-LSTM on full network
6. **Importance Scoring**: Aggregate attention weights per node
7. **Location Selection**: Top-N locations by importance score

## Key Functions
- `load_data_safely()` – Handles headerless/header CSV files
- `SpatioTemporalDataset` – PyTorch dataset for sequence data
- `GAT_LSTM` – Combined spatial-temporal model with attention

## Validation
Re-run the script using `GNN_optimal_network_data.csv` as input to compare reduced network performance against the full network baseline.

## Notes
- Graph constructed using haversine distance for geographic accuracy
- Attention weights indicate learned spatial dependencies
- Higher importance scores = more informative locations for prediction


