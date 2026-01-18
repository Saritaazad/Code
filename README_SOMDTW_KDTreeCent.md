# SOM-DTW with KDTree Centroid Selection

## Overview
Implements Self-Organizing Maps (SOM) for clustering precipitation time series data, combined with KDTree-based spatial queries to identify cluster centroids and generate rainfall predictions for ungauged locations in the Northwestern Himalayas.

## Purpose
Groups grid locations with similar rainfall patterns into homogeneous clusters using neural network-based pattern recognition, then uses spatial indexing to enable rapid nearest-neighbor queries for rainfall prediction at arbitrary coordinates.

## Methodology
1. **SOM Clustering**: 8×8 Self-Organizing Map trained on 42-year daily precipitation time series
2. **Pattern Assignment**: Each grid point assigned to winning SOM node based on time series similarity
3. **Centroid Extraction**: Geographic centroid calculated for each cluster
4. **KDTree Indexing**: Spatial index built for fast nearest-neighbor queries
5. **Prediction**: New locations mapped to nearest cluster for rainfall estimation

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `som_x`, `som_y` | 8 | SOM grid dimensions (64 total nodes) |
| `sigma` | 0.3 | SOM neighborhood radius |
| `learning_rate` | 0.1 | SOM learning rate |
| `epochs` | 1500 | Training iterations |

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
minisom
tslearn
geopandas
```

## Input Files
- `TimeSeriesDataHP.csv` – Gridded precipitation time series (Lat, Lon, daily values)
- `NWH/4-17-2018-899072.shp` – Northwestern Himalayas shapefile for visualization

## Output Files
| File | Description |
|------|-------------|
| `centroid64.csv` | Cluster centroid coordinates (up to 64 clusters) |
| `pred_{lat}_{lon}.csv` | Rainfall predictions for specified test locations |

## Key Functions
- `plot_som_series_averaged_center()` – Visualizes cluster prototypes with member series
- `generate_prediction(lat, lon)` – Returns rainfall time series for any coordinate
- `is_valid_number()` – Validates numeric values (handles NaN)

## Workflow
```
Load Data → Train SOM → Assign Clusters → Calculate Centroids → Build KDTree → Generate Predictions
```

## Test Locations
Pre-configured test points include IMD AWS station coordinates for validation against observed data.

## Author
Developed for PhD research on optimal rain gauge network design using SOM-DTW methodology in the Northwestern Himalayas.
