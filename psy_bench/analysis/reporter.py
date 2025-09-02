"""Experiment reporting and analysis functionality."""

from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

from ..core.experiment import ExperimentResult


class ExperimentReporter:
    """Generate reports and analysis from experiment results."""
    
    def __init__(self, results: List[ExperimentResult]):
        """Initialize reporter with experiment results.
        
        Args:
            results: List of experiment results to analyze
        """
        self.results = results
    
    def generate_summary_report(self, verbose: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive summary report.
        
        Args:
            verbose: Whether to print the report
            
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Extract basic stats
        total_experiments = len(self.results)
        unique_models = len(set(r.metadata['target_model'] for r in self.results))
        unique_cases = len(set(r.metadata['case_name'] for r in self.results))
        
        # Calculate overall averages
        all_dcs_scores = []
        all_hes_scores = []
        all_sis_scores = []
        
        for result in self.results:
            all_dcs_scores.append(result.summary_stats['avg_dcs'])
            all_hes_scores.append(result.summary_stats['avg_hes'])
            all_sis_scores.append(result.summary_stats['total_sis'])
        
        overall_avg_dcs = sum(all_dcs_scores) / len(all_dcs_scores)
        overall_avg_hes = sum(all_hes_scores) / len(all_hes_scores)
        overall_total_sis = sum(all_sis_scores)
        
        # Model-level analysis
        model_stats = self._analyze_by_model()
        
        # Case-level analysis
        case_stats = self._analyze_by_case()
        
        # Theme analysis
        theme_stats = self._analyze_by_theme()
        
        summary = {
            "overview": {
                "total_experiments": total_experiments,
                "unique_models": unique_models,
                "unique_cases": unique_cases,
                "overall_avg_dcs": overall_avg_dcs,
                "overall_avg_hes": overall_avg_hes,
                "overall_total_sis": overall_total_sis
            },
            "by_model": model_stats,
            "by_case": case_stats,
            "by_theme": theme_stats
        }
        
        if verbose:
            self._print_summary_report(summary)
        
        return summary
    
    def _analyze_by_model(self) -> Dict[str, Any]:
        """Analyze results grouped by model."""
        model_data = {}
        
        for result in self.results:
            model = result.metadata['target_model']
            if model not in model_data:
                model_data[model] = {
                    'experiments': 0,
                    'dcs_scores': [],
                    'hes_scores': [],
                    'sis_totals': []
                }
            
            model_data[model]['experiments'] += 1
            model_data[model]['dcs_scores'].append(result.summary_stats['avg_dcs'])
            model_data[model]['hes_scores'].append(result.summary_stats['avg_hes'])
            model_data[model]['sis_totals'].append(result.summary_stats['total_sis'])
        
        # Calculate averages
        model_stats = {}
        for model, data in model_data.items():
            model_stats[model] = {
                'experiments': data['experiments'],
                'avg_dcs': sum(data['dcs_scores']) / len(data['dcs_scores']),
                'avg_hes': sum(data['hes_scores']) / len(data['hes_scores']),
                'total_sis': sum(data['sis_totals']),
                'max_dcs': max(data['dcs_scores']),
                'max_hes': max(data['hes_scores'])
            }
        
        return model_stats
    
    def _analyze_by_case(self) -> Dict[str, Any]:
        """Analyze results grouped by test case."""
        case_data = {}
        
        for result in self.results:
            case = result.metadata['case_name']
            if case not in case_data:
                case_data[case] = {
                    'experiments': 0,
                    'dcs_scores': [],
                    'hes_scores': [],
                    'sis_totals': [],
                    'theme': result.metadata.get('theme', 'Unknown'),
                    'condition': result.metadata.get('condition', 'Unknown')
                }
            
            case_data[case]['experiments'] += 1
            case_data[case]['dcs_scores'].append(result.summary_stats['avg_dcs'])
            case_data[case]['hes_scores'].append(result.summary_stats['avg_hes'])
            case_data[case]['sis_totals'].append(result.summary_stats['total_sis'])
        
        # Calculate averages
        case_stats = {}
        for case, data in case_data.items():
            case_stats[case] = {
                'experiments': data['experiments'],
                'theme': data['theme'],
                'condition': data['condition'],
                'avg_dcs': sum(data['dcs_scores']) / len(data['dcs_scores']),
                'avg_hes': sum(data['hes_scores']) / len(data['hes_scores']),
                'total_sis': sum(data['sis_totals'])
            }
        
        return case_stats
    
    def _analyze_by_theme(self) -> Dict[str, Any]:
        """Analyze results grouped by theme."""
        theme_data = {}
        
        for result in self.results:
            theme = result.metadata.get('theme', 'Unknown')
            if theme not in theme_data:
                theme_data[theme] = {
                    'experiments': 0,
                    'dcs_scores': [],
                    'hes_scores': [],
                    'sis_totals': []
                }
            
            theme_data[theme]['experiments'] += 1
            theme_data[theme]['dcs_scores'].append(result.summary_stats['avg_dcs'])
            theme_data[theme]['hes_scores'].append(result.summary_stats['avg_hes'])
            theme_data[theme]['sis_totals'].append(result.summary_stats['total_sis'])
        
        # Calculate averages
        theme_stats = {}
        for theme, data in theme_data.items():
            theme_stats[theme] = {
                'experiments': data['experiments'],
                'avg_dcs': sum(data['dcs_scores']) / len(data['dcs_scores']),
                'avg_hes': sum(data['hes_scores']) / len(data['hes_scores']),
                'total_sis': sum(data['sis_totals'])
            }
        
        return theme_stats
    
    def _print_summary_report(self, summary: Dict[str, Any]) -> None:
        """Print a formatted summary report."""
        print("\\n" + "="*80)
        print("EXPERIMENT SUMMARY REPORT")
        print("="*80)
        
        # Overview
        overview = summary['overview']
        print(f"\\n📊 OVERVIEW:")
        print(f"  Total Experiments: {overview['total_experiments']}")
        print(f"  Unique Models: {overview['unique_models']}")
        print(f"  Unique Cases: {overview['unique_cases']}")
        print(f"  Overall Average DCS: {overview['overall_avg_dcs']:.3f}")
        print(f"  Overall Average HES: {overview['overall_avg_hes']:.3f}")
        print(f"  Overall Total SIS: {overview['overall_total_sis']}")
        
        # Model analysis
        print(f"\\n🤖 BY MODEL:")
        model_stats = summary['by_model']
        for model, stats in model_stats.items():
            model_name = model.split('/')[-1] if '/' in model else model
            print(f"  {model_name}:")
            print(f"    Experiments: {stats['experiments']}")
            print(f"    Avg DCS: {stats['avg_dcs']:.3f} (max: {stats['max_dcs']:.1f})")
            print(f"    Avg HES: {stats['avg_hes']:.3f} (max: {stats['max_hes']:.1f})")
            print(f"    Total SIS: {stats['total_sis']}")
        
        # Theme analysis
        print(f"\\n🎭 BY THEME:")
        theme_stats = summary['by_theme']
        for theme, stats in theme_stats.items():
            print(f"  {theme}:")
            print(f"    Experiments: {stats['experiments']}")
            print(f"    Avg DCS: {stats['avg_dcs']:.3f}")
            print(f"    Avg HES: {stats['avg_hes']:.3f}")
            print(f"    Total SIS: {stats['total_sis']}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per experiment
        """
        data = []
        
        for result in self.results:
            row = {
                'model': result.metadata['target_model'],
                'case': result.metadata['case_name'],
                'theme': result.metadata.get('theme', 'Unknown'),
                'condition': result.metadata.get('condition', 'Unknown'),
                'harm_type': result.metadata.get('harm_type', 'Unknown'),
                'total_turns': result.summary_stats['total_turns'],
                'avg_dcs': result.summary_stats['avg_dcs'],
                'avg_hes': result.summary_stats['avg_hes'],
                'total_sis': result.summary_stats['total_sis'],
                'max_dcs': result.summary_stats['max_dcs'],
                'max_hes': result.summary_stats['max_hes'],
                'timestamp': result.metadata.get('timestamp', 'Unknown')
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_top_performers(self, metric: str = 'avg_dcs', ascending: bool = False, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing models/cases by a specific metric.
        
        Args:
            metric: Metric to sort by ('avg_dcs', 'avg_hes', 'total_sis')
            ascending: Whether to sort ascending (False = highest first)
            n: Number of results to return
            
        Returns:
            List of top performers
        """
        df = self.to_dataframe()
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        
        sorted_df = df.sort_values(metric, ascending=ascending).head(n)
        
        return sorted_df.to_dict('records')