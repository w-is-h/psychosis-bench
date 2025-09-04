# Visualization Helper Script

The `visualize_results.py` script provides a convenient way to generate interactive visualizations from CSV files or regenerate visualizations from existing output folders.

## Features

- **CSV Input**: Generate visualizations directly from CSV files containing experiment results
- **Folder Regeneration**: Regenerate visualizations from existing output folders
- **Updated Color Scaling**: Properly scaled for the new 0-2 (DCS/HES) and 0-1 (SIS) scoring ranges
- **Interactive HTML**: Creates interactive Plotly visualizations
- **Styled Tables**: Generates color-coded HTML tables with proper scaling

## Usage

### Generate from CSV File

```bash
# Basic usage
python visualize_results.py --csv results.csv --output visualizations/

# Use default output directory
python visualize_results.py --csv results.csv
```

### Regenerate from Output Folder

```bash
# Regenerate visualizations in existing folder
python visualize_results.py --regenerate async_batch_outputs/
```

## Generated Visualizations

The script creates the following interactive HTML files:

1. **`model_comparison.html`** - Bar chart comparing model performance
2. **`theme_analysis.html`** - Radar chart showing performance by theme (if multiple themes)
3. **`score_distributions.html`** - Histograms of DCS and HES score distributions
4. **`safety_landscape.html`** - Scatter plot of DCS vs HES with SIS as bubble size
5. **`styled_statistics_table.html`** - Color-coded statistics table
6. **`case_performance_table.html`** - Per-case performance across all models

## Color Coding

- **DCS/HES Scores**: Green (0) → Yellow (1) → Orange (1.5) → Red (2) - Lower is better
- **SIS Scores**: Green (0) → Yellow (0.5) → Orange (0.75) → Red (1) - Higher is better

## CSV Format Requirements

The CSV file should contain the following columns:
- `model`: Model name
- `case`: Test case name
- `theme`: Case theme
- `condition`: Explicit/Implicit
- `harm_type`: Type of harm
- `total_turns`: Number of turns
- `avg_dcs`: Average DCS score (0-2)
- `avg_hes`: Average HES score (0-2)
- `total_sis`: Total SIS score (0-1 per turn, summed)
- `max_dcs`: Maximum DCS score (optional)
- `max_hes`: Maximum HES score (optional)

## Examples

```bash
# Generate from your own CSV
python visualize_results.py --csv my_results.csv --output my_analysis/

# Regenerate existing visualizations
python visualize_results.py --regenerate async_batch_outputs/

# Quick test with existing data
python visualize_results.py --csv async_batch_outputs/async_results.csv
```

## Dependencies

The script requires the same dependencies as the main psy-bench library:
- pandas
- plotly
- numpy
- psy-bench (the main library)

## Notes

- The script creates fake `ExperimentResult` objects from CSV data to maintain compatibility with the existing analyzer
- All visualizations use the updated 0-2 and 0-1 scoring ranges
- Color scaling is automatically adjusted for the new ranges
- The script preserves the original functionality while being more flexible for different data sources
