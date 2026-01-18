# Himachal Pradesh Gridded Data Extraction

## Overview
Data preprocessing script that extracts time series meteorological data from NASA POWER gridded datasets, parsing encoded column headers to recover geographic coordinates and filtering for the Himachal Pradesh region.

## Purpose
Converts raw gridded climate data with encoded header formats into structured CSV files with explicit latitude/longitude coordinates, handling missing values and geographic subsetting for the Northwestern Himalayas study area.

## Input
`Daily_Data_42_Years_1981-2022.csv` – NASA POWER daily meteorological data with column headers encoding location information in underscore-separated format.

## Processing Steps
1. Parses column headers to extract encoded longitude/latitude values
2. Converts coordinate strings to decimal degrees (e.g., `7532` → `75.32`)
3. Replaces invalid data flags (< -900) with zero
4. Filters grid points for HP region: Lat 30–34°N, Lon 75–80°E
5. Exports filtered and complete datasets

## Output Files
| File | Description |
|------|-------------|
| `TimeSeriesDataHP.csv` | Grid points within Himachal Pradesh bounds |
| `TimeSeriesDataALL.csv` | All gridded locations (full domain) |

## Output Format
Each row contains:
```
Latitude, Longitude, Day1_Value, Day2_Value, ..., DayN_Value
```

## Requirements
```
pandas
csv (standard library)
```

## Usage
```python
python extDataHP.py
```

## Notes
- Processes columns at intervals of 7 (multi-variable gridded structure)
- Coordinate parsing handles 3-digit and 4-digit encoded formats
- Missing/invalid values (NASA POWER flag: -999) replaced with 0

## Author
Developed for PhD research on precipitation monitoring in the Northwestern Himalayas.
