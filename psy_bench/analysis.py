"""Unified analysis and export module."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .core.models import ExperimentResult, ScoreType


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class ResultAnalyzer:
    """Analyze and export experiment results."""
    
    def __init__(self, results: List[ExperimentResult]):
        """Initialize analyzer with results.
        
        Args:
            results: List of experiment results
        """
        self.results = results
        self._summary = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary.
        
        Returns:
            Dictionary with analysis results
        """
        if self._summary is None:
            self._summary = self._calculate_summary()
        return self._summary
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate all summary statistics."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic metrics
        total_experiments = len(self.results)
        unique_models = set(r.model for r in self.results)
        unique_cases = set(r.test_case.name for r in self.results)
        
        # Calculate overall averages
        all_summaries = [r.summary for r in self.results]
        
        overall_stats = {
            "total_experiments": total_experiments,
            "unique_models": len(unique_models),
            "unique_cases": len(unique_cases),
            "avg_dcs": sum(s["avg_dcs"] for s in all_summaries) / len(all_summaries),
            "avg_hes": sum(s["avg_hes"] for s in all_summaries) / len(all_summaries),
            "total_sis": sum(s["total_sis"] for s in all_summaries),
        }
        
        return {
            "overview": overall_stats,
            "by_model": self._analyze_by_model(),
            "by_case": self._analyze_by_case(),
            "by_theme": self._analyze_by_theme()
        }
    
    def _analyze_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by model."""
        model_groups = {}
        
        for result in self.results:
            model = result.model
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result.summary)
        
        model_stats = {}
        for model, summaries in model_groups.items():
            model_stats[model] = {
                "experiments": len(summaries),
                "avg_dcs": sum(s["avg_dcs"] for s in summaries) / len(summaries),
                "avg_hes": sum(s["avg_hes"] for s in summaries) / len(summaries),
                "total_sis": sum(s["total_sis"] for s in summaries),
                "max_dcs": max(s["max_dcs"] for s in summaries),
                "max_hes": max(s["max_hes"] for s in summaries),
            }
        
        return model_stats
    
    def _analyze_by_case(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by test case."""
        case_groups = {}
        
        for result in self.results:
            case_name = result.test_case.name
            if case_name not in case_groups:
                case_groups[case_name] = {
                    "summaries": [],
                    "theme": result.test_case.theme,
                    "condition": result.test_case.condition
                }
            case_groups[case_name]["summaries"].append(result.summary)
        
        case_stats = {}
        for case_name, data in case_groups.items():
            summaries = data["summaries"]
            case_stats[case_name] = {
                "experiments": len(summaries),
                "theme": data["theme"],
                "condition": data["condition"],
                "avg_dcs": sum(s["avg_dcs"] for s in summaries) / len(summaries),
                "avg_hes": sum(s["avg_hes"] for s in summaries) / len(summaries),
                "total_sis": sum(s["total_sis"] for s in summaries),
            }
        
        return case_stats
    
    def _analyze_by_theme(self) -> Dict[str, Dict[str, Any]]:
        """Group and analyze results by theme."""
        theme_groups = {}
        
        for result in self.results:
            theme = result.test_case.theme
            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(result.summary)
        
        theme_stats = {}
        for theme, summaries in theme_groups.items():
            theme_stats[theme] = {
                "experiments": len(summaries),
                "avg_dcs": sum(s["avg_dcs"] for s in summaries) / len(summaries),
                "avg_hes": sum(s["avg_hes"] for s in summaries) / len(summaries),
                "total_sis": sum(s["total_sis"] for s in summaries),
            }
        
        return theme_stats
    
    def print_summary(self, detailed: bool = True):
        """Print formatted summary to console.
        
        Args:
            detailed: Whether to include detailed breakdowns
        """
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EXPERIMENT ANALYSIS SUMMARY")
        print("="*60)
        
        # Overview
        overview = summary["overview"]
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Experiments: {overview['total_experiments']}")
        print(f"  Unique Models: {overview['unique_models']}")
        print(f"  Unique Cases: {overview['unique_cases']}")
        print(f"  Overall Avg DCS: {overview['avg_dcs']:.3f}")
        print(f"  Overall Avg HES: {overview['avg_hes']:.3f}")
        print(f"  Overall Total SIS: {overview['total_sis']}")
        
        if detailed:
            # Model breakdown
            print(f"\n🤖 BY MODEL:")
            for model, stats in summary["by_model"].items():
                print(f"\n  {model}:")
                print(f"    Experiments: {stats['experiments']}")
                print(f"    Avg DCS: {stats['avg_dcs']:.3f} (max: {stats['max_dcs']})")
                print(f"    Avg HES: {stats['avg_hes']:.3f} (max: {stats['max_hes']})")
                print(f"    Total SIS: {stats['total_sis']}")
            
            # Theme breakdown
            print(f"\n🎭 BY THEME:")
            for theme, stats in summary["by_theme"].items():
                print(f"\n  {theme}:")
                print(f"    Experiments: {stats['experiments']}")
                print(f"    Avg DCS: {stats['avg_dcs']:.3f}")
                print(f"    Avg HES: {stats['avg_hes']:.3f}")
                print(f"    Total SIS: {stats['total_sis']}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with experiment data
        """
        rows = []
        
        for result in self.results:
            summary = result.summary
            row = {
                "model": result.model,
                "case": result.test_case.name,
                "theme": result.test_case.theme,
                "condition": result.test_case.condition,
                "harm_type": result.test_case.harm_type,
                "total_turns": summary["total_turns"],
                "avg_dcs": summary["avg_dcs"],
                "avg_hes": summary["avg_hes"],
                "total_sis": summary["total_sis"],
                "max_dcs": summary["max_dcs"],
                "max_hes": summary["max_hes"],
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export(
        self,
        format: ExportFormat | str,
        path: Union[str, Path],
        **options
    ) -> None:
        """Export results in specified format.
        
        Args:
            format: Export format
            path: Output file path
            **options: Format-specific options
        """
        # Normalize format to ExportFormat if string provided
        if isinstance(format, str):
            try:
                format = ExportFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unsupported format: {format}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ExportFormat.JSON:
            self._export_json(path, **options)
        elif format == ExportFormat.CSV:
            self._export_csv(path, **options)
        elif format == ExportFormat.EXCEL:
            self._export_excel(path, **options)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Exported {len(self.results)} results to {path}")
    
    def _export_json(self, path: Path, indent: int = 2, **kwargs):
        """Export to JSON format."""
        # Ensure enum keys in Turn.scores are JSON-friendly (use enum values)
        def _normalize_scores(result):
            data = result.model_dump()
            for turn in data.get("turns", []):
                scores = turn.get("scores")
                if isinstance(scores, dict):
                    turn["scores"] = {
                        (k.value if hasattr(k, "value") else str(k)): v
                        for k, v in scores.items()
                    }
            return data

        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_experiments": len(self.results),
            },
            "summary": self.get_summary(),
            "results": [_normalize_scores(result) for result in self.results]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False, **kwargs)
    
    def _export_csv(self, path: Path, **kwargs):
        """Export to CSV format."""
        df = self.to_dataframe()
        df.to_csv(path, index=False, **kwargs)
    
    def _export_excel(self, path: Path, include_details: bool = True, **kwargs):
        """Export to Excel with multiple sheets."""
        with pd.ExcelWriter(path, engine='openpyxl', **kwargs) as writer:
            # Summary sheet
            df = self.to_dataframe()
            df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Model comparison
            model_stats = self.get_summary()["by_model"]
            model_df = pd.DataFrame.from_dict(model_stats, orient='index')
            model_df.to_excel(writer, sheet_name='By Model')
            
            # Theme analysis
            theme_stats = self.get_summary()["by_theme"]
            theme_df = pd.DataFrame.from_dict(theme_stats, orient='index')
            theme_df.to_excel(writer, sheet_name='By Theme')
            
            # Case analysis
            if include_details:
                case_stats = self.get_summary()["by_case"]
                case_df = pd.DataFrame.from_dict(case_stats, orient='index')
                case_df.to_excel(writer, sheet_name='By Case')
