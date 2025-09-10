#!/usr/bin/env python3
"""Helper script to generate visualizations from CSV files or existing output folders."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.colors import qualitative, sequential, diverging
import colorsys

from psy_bench.analysis import ResultAnalyzer, ExportFormat
from psy_bench.core.models import ExperimentResult, TestCase, Turn, ScoreType


def load_csv_to_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV file and return DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} rows from {csv_path}")
    return df


class CSVAnalyzer:
    """Analyzer that works directly with CSV data without reconstructing individual turns."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = df
        self._summary = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        if self._summary is None:
            self._summary = self._calculate_summary()
        return self._summary
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate all summary statistics."""
        if self.df.empty:
            return {"error": "No data to analyze"}
        
        # Basic metrics
        total_experiments = len(self.df)
        unique_models = self.df['model'].unique()
        unique_cases = self.df['case'].unique()
        
        # Calculate overall averages
        overall_stats = {
            "total_experiments": total_experiments,
            "unique_models": len(unique_models),
            "unique_cases": len(unique_cases),
            "avg_dcs": self.df['avg_dcs'].mean(),
            "avg_hes": self.df['avg_hes'].mean(),
            "total_sis": self.df['total_sis'].sum(),
        }
        
        return {
            "overview": overall_stats,
            "by_model": self._analyze_by_model(),
            "by_case": self._analyze_by_case(),
            "by_theme": self._analyze_by_theme()
        }
    
    def _analyze_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by model."""
        model_stats = {}
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            model_stats[model] = {
                "experiments": len(model_data),
                "avg_dcs": model_data['avg_dcs'].mean(),
                "avg_hes": model_data['avg_hes'].mean(),
                "total_sis": model_data['total_sis'].sum(),
                "max_dcs": model_data['max_dcs'].max(),
                "max_hes": model_data['max_hes'].max(),
            }
        
        return model_stats
    
    def _analyze_by_case(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by test case."""
        case_stats = {}
        
        for case in self.df['case'].unique():
            case_data = self.df[self.df['case'] == case]
            case_stats[case] = {
                "experiments": len(case_data),
                "theme": case_data['theme'].iloc[0],
                "condition": case_data['condition'].iloc[0],
                "avg_dcs": case_data['avg_dcs'].mean(),
                "avg_hes": case_data['avg_hes'].mean(),
                "total_sis": case_data['total_sis'].sum(),
            }
        
        return case_stats
    
    def _analyze_by_theme(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by theme."""
        theme_stats = {}
        
        for theme in self.df['theme'].unique():
            theme_data = self.df[self.df['theme'] == theme]
            theme_stats[theme] = {
                "experiments": len(theme_data),
                "avg_dcs": theme_data['avg_dcs'].mean(),
                "avg_hes": theme_data['avg_hes'].mean(),
                "total_sis": theme_data['total_sis'].sum(),
            }
        
        return theme_stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return the original DataFrame."""
        return self.df.copy()


def create_fake_experiment_results(df: pd.DataFrame) -> List[ExperimentResult]:
    """Convert DataFrame to list of ExperimentResult objects for compatibility with analyzer."""
    # This function is no longer used, but kept for backward compatibility
    # The CSVAnalyzer class above handles CSV data directly
    return []


def create_styled_table_html(df):
    """Create a styled HTML table with background gradients without requiring jinja2."""
    
    # Get numeric columns (exclude the first column which is usually the index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate min and max for each numeric column (fallbacks for unknown metrics)
    col_mins = df[numeric_cols].min()
    col_maxs = df[numeric_cols].max()

    def determine_metric_kind(col_name: str, row) -> str | None:
        # Prefer row-wise score_type if present (e.g., 'avg_dcs', 'avg_hes', 'total_sis')
        if 'score_type' in df.columns:
            try:
                st = str(row['score_type']).lower()
                if 'dcs' in st:
                    return 'dcs'
                if 'hes' in st:
                    return 'hes'
                if 'sis' in st:
                    return 'sis'
            except Exception:
                pass
        # Otherwise infer from column name (used in case-performance table)
        name = str(col_name).lower()
        if 'dcs' in name:
            return 'dcs'
        if 'hes' in name:
            return 'hes'
        if 'sis' in name:
            return 'sis'
        return None

    def to_rgb_css_from_normalized(norm: float, invert: bool) -> tuple[str, str]:
        # Clamp
        norm = 0.0 if np.isnan(norm) else max(0.0, min(1.0, float(norm)))
        # Hue: green (120°) to red (0°)
        hue_deg = (120.0 * norm) if invert else (120.0 * (1.0 - norm))
        h = hue_deg / 360.0
        l = 0.5
        s = 0.7
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        R, G, B = int(r * 255), int(g * 255), int(b * 255)
        bg = f'rgb({R}, {G}, {B})'
        # Perceived brightness threshold for text color
        text = 'white' if (R * 0.299 + G * 0.587 + B * 0.114) < 160 else 'black'
        return bg, text
    
    # Create HTML table
    html = ['<table>']
    
    # Add header row
    html.append('<thead><tr>')
    for col in df.columns:
        html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')
    
    # Add data rows
    html.append('<tbody>')
    for _, row in df.iterrows():
        html.append('<tr>')
        # Precompute row-level SIS normalization reference if needed
        row_metric_kind = None
        if 'score_type' in df.columns:
            try:
                st = str(row['score_type']).lower()
                if 'sis' in st:
                    row_metric_kind = 'sis'
            except Exception:
                row_metric_kind = None
        sis_row_ref = None
        if row_metric_kind == 'sis':
            # Use row-wise robust reference for SIS if total_turns not available
            sis_values = []
            for c, v in row.items():
                if c in numeric_cols and ('mean' in str(c).lower() or 'sis' in str(c).lower()):
                    try:
                        sis_values.append(float(v))
                    except Exception:
                        pass
            if sis_values:
                sis_row_ref = max(np.percentile(sis_values, 95), max(sis_values))

        for i, (col, value) in enumerate(row.items()):
            if col in numeric_cols:
                metric_kind = determine_metric_kind(col, row)
                # In pivoted statistics table, color only 'mean' columns
                if 'score_type' in df.columns and ('mean' not in str(col).lower()):
                    html.append(f'<td>{value:.3f}</td>')
                    continue

                if metric_kind == 'dcs' or metric_kind == 'hes':
                    normalized = float(value) / 2.0 if value is not None else 0.5
                    bg_color, text_color = to_rgb_css_from_normalized(normalized, invert=False)
                    html.append(f'<td style="background-color: {bg_color}; color: {text_color};">{value:.3f}</td>')
                elif metric_kind == 'sis':
                    # Prefer absolute normalization if total_turns available
                    denom = None
                    if 'total_turns' in df.columns:
                        try:
                            denom = 1.0 * float(row['total_turns'])  # SIS is 0-1 per turn
                        except Exception:
                            denom = None
                    if denom is None:
                        # Fallback to row-wise robust reference if available; else column max-min
                        if sis_row_ref and sis_row_ref > 0:
                            denom = sis_row_ref
                        else:
                            rng = (col_maxs[col] - col_mins[col])
                            denom = float(col_maxs[col]) if rng == 0 else float(col_maxs[col])
                            if denom <= 0:
                                denom = 1.0
                    normalized = float(value) / denom if denom else 0.5
                    bg_color, text_color = to_rgb_css_from_normalized(normalized, invert=True)
                    html.append(f'<td style="background-color: {bg_color}; color: {text_color};">{value:.3f}</td>')
                else:
                    # Unrecognized metric; leave uncolored to avoid misleading cues
                    html.append(f'<td>{value:.3f}</td>')
            else:
                html.append(f'<td>{value}</td>')
        html.append('</tr>')
    html.append('</tbody>')
    
    html.append('</table>')
    return '\n'.join(html)


def create_case_performance_table_html(case_performance_data, unique_models):
    """Create a case performance table with model names as merged headers and DCS/HES/SIS as sub-columns."""
    
    def determine_metric_kind(col_suffix: str) -> str:
        if 'DCS' in col_suffix:
            return 'dcs'
        elif 'HES' in col_suffix:
            return 'hes'
        elif 'SIS' in col_suffix:
            return 'sis'
        return 'unknown'

    def to_rgb_css_from_normalized(norm: float, invert: bool) -> tuple[str, str]:
        # Clamp
        norm = 0.0 if np.isnan(norm) else max(0.0, min(1.0, float(norm)))
        # Hue: green (120°) to red (0°)
        hue_deg = (120.0 * norm) if invert else (120.0 * (1.0 - norm))
        h = hue_deg / 360.0
        l = 0.5
        s = 0.7
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        R, G, B = int(r * 255), int(g * 255), int(b * 255)
        bg = f'rgb({R}, {G}, {B})'
        # Perceived brightness threshold for text color
        text = 'white' if (R * 0.299 + G * 0.587 + B * 0.114) < 160 else 'black'
        return bg, text
    
    html = ['<table>']
    
    # Create header rows
    html.append('<thead>')
    
    # Top header row with model names (merged)
    html.append('<tr>')
    html.append('<th rowspan="2" class="case-name">Case</th>')  # Case column spans 2 rows
    for model in unique_models:
        html.append(f'<th colspan="3" class="model-header">{model}</th>')  # Each model spans 3 columns
    html.append('</tr>')
    
    # Sub-header row with DCS/HES/SIS
    html.append('<tr>')
    for model in unique_models:
        html.append('<th class="score-header">DCS</th>')
        html.append('<th class="score-header">HES</th>')
        html.append('<th class="score-header">SIS</th>')
    html.append('</tr>')
    
    html.append('</thead>')
    
    # Add data rows
    html.append('<tbody>')
    for case_data in case_performance_data:
        html.append('<tr>')
        html.append(f'<td class="case-name">{case_data["Case"]}</td>')
        
        for model in unique_models:
            dcs_value = case_data.get(f'{model}_DCS', 0)
            hes_value = case_data.get(f'{model}_HES', 0)
            sis_value = case_data.get(f'{model}_SIS', 0)
            
            # DCS cell (lower is better, so 0 should be green)
            normalized = float(dcs_value) / 2.0
            bg_color, text_color = to_rgb_css_from_normalized(normalized, invert=False)
            html.append(f'<td style="background-color: {bg_color}; color: {text_color};">{dcs_value:.3f}</td>')
            
            # HES cell (lower is better, so 0 should be green)
            normalized = float(hes_value) / 2.0
            bg_color, text_color = to_rgb_css_from_normalized(normalized, invert=False)
            html.append(f'<td style="background-color: {bg_color}; color: {text_color};">{hes_value:.3f}</td>')
            
            # SIS cell (higher is better for interventions, so 0 should be red)
            # Assume max reasonable SIS is around 5 for normalization (adjust as needed)
            normalized = min(float(sis_value) / 5.0, 1.0)
            bg_color, text_color = to_rgb_css_from_normalized(normalized, invert=True)
            html.append(f'<td style="background-color: {bg_color}; color: {text_color};">{sis_value:.3f}</td>')
        
        html.append('</tr>')
    html.append('</tbody>')
    
    html.append('</table>')
    return '\n'.join(html)


def create_visualizations(analyzer, output_dir: Path) -> int:
    """Create interactive visualizations using plotly."""
    print("📊 Creating interactive visualizations...")
    
    # Get summary data
    summary = analyzer.get_summary()
    df = analyzer.to_dataframe()
    
    # 1. Model Performance Comparison
    model_stats = []
    for model, stats in summary["by_model"].items():
        model_stats.append({
            'Model': model.replace('/', '<br>'),  # Line break for better display
            'Avg DCS': stats['avg_dcs'],
            'Avg HES': stats['avg_hes'],
            'Total SIS': stats['total_sis'],
            'Experiments': stats['experiments']
        })
    
    if model_stats:
        # Model performance bar chart
        fig_models = go.Figure()
        
        models = [stat['Model'] for stat in model_stats]
        dcs_scores = [stat['Avg DCS'] for stat in model_stats]
        hes_scores = [stat['Avg HES'] for stat in model_stats]
        
        fig_models.add_trace(go.Bar(
            name='Avg DCS (Lower = Better)',
            x=models,
            y=dcs_scores,
            marker_color='lightcoral',
            text=[f'{score:.2f}' for score in dcs_scores],
            textposition='auto',
        ))
        
        fig_models.add_trace(go.Bar(
            name='Avg HES (Lower = Better)',
            x=models,
            y=hes_scores,
            marker_color='lightblue',
            text=[f'{score:.2f}' for score in hes_scores],
            textposition='auto',
        ))
        
        fig_models.update_layout(
            title='Model Safety Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white'
        )
        
        model_viz_path = output_dir / 'model_comparison.html'
        fig_models.write_html(str(model_viz_path))
        print(f"   📊 Model comparison: {model_viz_path}")
    
    # 2. Theme Analysis (if multiple themes)
    if len(summary["by_theme"]) > 1:
        theme_stats = []
        for theme, stats in summary["by_theme"].items():
            theme_stats.append({
                'Theme': theme,
                'Avg DCS': stats['avg_dcs'],
                'Avg HES': stats['avg_hes'],
                'Experiments': stats['experiments']
            })
        
        # Theme performance radar chart
        fig_themes = go.Figure()
        
        themes = [stat['Theme'] for stat in theme_stats]
        fig_themes.add_trace(go.Scatterpolar(
            r=[stat['Avg DCS'] for stat in theme_stats],
            theta=themes,
            fill='toself',
            name='DCS Scores',
            line_color='red'
        ))
        
        fig_themes.add_trace(go.Scatterpolar(
            r=[stat['Avg HES'] for stat in theme_stats],
            theta=themes,
            fill='toself',
            name='HES Scores',
            line_color='blue'
        ))
        
        fig_themes.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2]
                )),
            showlegend=True,
            title="Theme Analysis: Safety Scores by Theme"
        )
        
        theme_viz_path = output_dir / 'theme_analysis.html'
        fig_themes.write_html(str(theme_viz_path))
        print(f"   🎭 Theme analysis: {theme_viz_path}")
    
    # 3. Safety Score Distribution
    if len(df) > 1:
        fig_dist = make_subplots(
            rows=1, cols=2,
            subplot_titles=('DCS Distribution', 'HES Distribution')
        )
        
        # DCS histogram
        fig_dist.add_trace(
            go.Histogram(x=df['avg_dcs'], name='DCS', nbinsx=6, marker_color='lightcoral'),
            row=1, col=1
        )
        
        # HES histogram
        fig_dist.add_trace(
            go.Histogram(x=df['avg_hes'], name='HES', nbinsx=6, marker_color='lightblue'),
            row=1, col=2
        )
        
        fig_dist.update_layout(
            title_text="Safety Score Distributions",
            showlegend=False,
            template='plotly_white'
        )
        fig_dist.update_xaxes(title_text="DCS Score", row=1, col=1)
        fig_dist.update_xaxes(title_text="HES Score", row=1, col=2)
        
        dist_viz_path = output_dir / 'score_distributions.html'
        fig_dist.write_html(str(dist_viz_path))
        print(f"   📈 Score distributions: {dist_viz_path}")
    
    # 4. Safety Intervention Analysis
    if 'total_sis' in df.columns and df['total_sis'].sum() > 0:
        # Create a scatter plot of DCS vs HES with SIS as size
        fig_safety = px.scatter(
            df, 
            x='avg_dcs', 
            y='avg_hes',
            size='total_sis',
            color='model',
            hover_data=['case', 'total_sis'],
            title='Safety Landscape: DCS vs HES (Bubble size = Safety Interventions)',
            labels={'avg_dcs': 'Average DCS', 'avg_hes': 'Average HES'}
        )
        
        fig_safety.update_layout(template='plotly_white')
        
        safety_viz_path = output_dir / 'safety_landscape.html'
        fig_safety.write_html(str(safety_viz_path))
        print(f"   🛡️  Safety landscape: {safety_viz_path}")
    
    # 5. Styled Statistics Table
    if len(df) > 0:
        # Create a pivoted table with statistics for each model and score type
        # First, melt the dataframe to have score types as rows
        score_columns = ['avg_dcs', 'avg_hes', 'total_sis']
        melted_df = df.melt(
            id_vars=['model'], 
            value_vars=score_columns,
            var_name='score_type', 
            value_name='score_value'
        )
        
        # Create pivot table with statistics
        pivoted_df = melted_df.groupby(['model', 'score_type'])['score_value'].agg([
            'count', 'mean', 'std', 'min'
        ]).round(6)
        
        # Flatten the multi-level columns
        pivoted_df.columns = [f'{col}' for col in pivoted_df.columns]
        pivoted_df = pivoted_df.reset_index()
        
        # Pivot again to get models as columns and score types as rows
        final_pivot = pivoted_df.pivot_table(
            index='score_type',
            columns='model',
            values=['count', 'mean', 'std', 'min'],
            fill_value=0
        )
        
        # Flatten column names
        final_pivot.columns = [f'{col[1]}_{col[0]}' for col in final_pivot.columns]
        final_pivot = final_pivot.reset_index()
        
        # Create styled table HTML without requiring jinja2
        styled_table_html = create_styled_table_html(final_pivot)
        
        # Save styled table to HTML file
        table_viz_path = output_dir / 'styled_statistics_table.html'
        with open(table_viz_path, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Statistics Table</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Model Performance Statistics</h1>
    <p>This table shows count, mean, standard deviation, and minimum values for each model and score type.</p>
    {styled_table_html}
</body>
</html>
            """)
        
        print(f"   📋 Styled statistics table: {table_viz_path}")
    
    # 6. Per-Case Performance Table
    if len(df) > 0:
        # Create a table showing each case with all models and their scores
        case_performance_data = []
        
        # Get unique cases and models
        unique_cases = df['case'].unique()
        unique_models = df['model'].unique()
        
        for case in unique_cases:
            case_data = {'Case': case}
            case_df = df[df['case'] == case]
            
            for model in unique_models:
                model_data = case_df[case_df['model'] == model]
                if len(model_data) > 0:
                    # Add DCS, HES, SIS for this model
                    case_data[f'{model}_DCS'] = model_data['avg_dcs'].iloc[0]
                    case_data[f'{model}_HES'] = model_data['avg_hes'].iloc[0]
                    case_data[f'{model}_SIS'] = model_data['total_sis'].iloc[0]
                else:
                    # Fill with 0 if no data
                    case_data[f'{model}_DCS'] = 0
                    case_data[f'{model}_HES'] = 0
                    case_data[f'{model}_SIS'] = 0
            
            case_performance_data.append(case_data)
        
        # Create DataFrame
        case_performance_df = pd.DataFrame(case_performance_data)
        
        # Create styled table HTML using the new hierarchical header function
        case_styled_table_html = create_case_performance_table_html(case_performance_data, unique_models)
        
        # Save case performance table to HTML file
        case_table_viz_path = output_dir / 'case_performance_table.html'
        with open(case_table_viz_path, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Case Performance Table</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .case-name {{ 
            text-align: left; 
            font-weight: bold; 
            background-color: #f8f9fa;
            min-width: 200px;
        }}
        .model-header {{ 
            background-color: #4a90e2; 
            color: white;
            font-size: 16px;
            font-weight: bold;
        }}
        .score-header {{
            background-color: #7bb3f0;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Case Performance Across Models</h1>
    <p>This table shows DCS, HES, and SIS scores for each test case across all models.</p>
    <p><strong>Legend:</strong> DCS = Delusion Confirmation Score, HES = Harm Enablement Score, SIS = Safety Intervention Score</p>
    <p><strong>Color coding:</strong> DCS/HES: Green → Yellow → Orange → Red (0–2 scale, lower is better) | SIS: Green → Yellow → Orange → Red (0–1 per turn, summed across turns)</p>
    {case_styled_table_html}
</body>
</html>
            """)
        
        print(f"   📊 Case performance table: {case_table_viz_path}")
    
    return len([f for f in output_dir.glob('*.html')])


def regenerate_from_output_folder(output_folder: Path) -> int:
    """Regenerate visualizations from existing output folder."""
    print(f"🔄 Regenerating visualizations from {output_folder}")
    
    # Look for CSV files in the output folder
    csv_files = list(output_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {output_folder}")
    
    # Use the first CSV file found
    csv_file = csv_files[0]
    print(f"📊 Using CSV file: {csv_file}")
    
    # Load and process
    df = load_csv_to_dataframe(csv_file)
    analyzer = CSVAnalyzer(df)
    
    # Create visualizations
    num_viz = create_visualizations(analyzer, output_folder)
    
    print(f"✅ Regenerated {num_viz} visualizations in {output_folder}")
    return num_viz


def generate_from_csv(csv_path: Path, output_dir: Path) -> int:
    """Generate visualizations from CSV file."""
    print(f"📊 Generating visualizations from {csv_path}")
    
    # Load and process
    df = load_csv_to_dataframe(csv_path)
    analyzer = CSVAnalyzer(df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    num_viz = create_visualizations(analyzer, output_dir)
    
    print(f"✅ Generated {num_viz} visualizations in {output_dir}")
    return num_viz


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from CSV files or regenerate from output folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from CSV file
  python visualize_results.py --csv results.csv --output visualizations/
  
  # Regenerate from existing output folder
  python visualize_results.py --regenerate async_batch_outputs/
  
  # Generate with custom output name
  python visualize_results.py --csv data.csv --output my_analysis/
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=Path, help='Path to CSV file to visualize')
    group.add_argument('--regenerate', type=Path, help='Path to output folder to regenerate visualizations')
    
    parser.add_argument('--output', type=Path, help='Output directory (default: same as input or "visualizations/")')
    
    args = parser.parse_args()
    
    try:
        if args.csv:
            # Generate from CSV
            output_dir = args.output or Path("visualizations")
            num_viz = generate_from_csv(args.csv, output_dir)
            print(f"\n🌐 Open the .html files in {output_dir} to view the visualizations!")
            
        elif args.regenerate:
            # Regenerate from output folder
            num_viz = regenerate_from_output_folder(args.regenerate)
            print(f"\n🌐 Open the .html files in {args.regenerate} to view the visualizations!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
