"""Export experiment results to various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd

from ..core.experiment import ExperimentResult


class BaseExporter:
    """Base class for result exporters."""
    
    def __init__(self, results: List[ExperimentResult]):
        """Initialize exporter with results.
        
        Args:
            results: List of experiment results to export
        """
        self.results = results
    
    def export(self, file_path: Union[str, Path], **kwargs) -> None:
        """Export results to file.
        
        Args:
            file_path: Path to save the exported file
            **kwargs: Format-specific options
        """
        raise NotImplementedError


class JSONExporter(BaseExporter):
    """Export experiment results to JSON format."""
    
    def export(self, file_path: Union[str, Path], indent: int = 2, **kwargs) -> None:
        """Export results to JSON file.
        
        Args:
            file_path: Path to save JSON file
            indent: JSON indentation level
            **kwargs: Additional JSON dump options
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(self.results),
                "psy_bench_version": "0.1.0"
            },
            "results": [result.model_dump() for result in self.results]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False, **kwargs)
        
        print(f"✅ Exported {len(self.results)} results to {file_path}")


class CSVExporter(BaseExporter):
    """Export experiment results to CSV format."""
    
    def export(self, file_path: Union[str, Path], flatten_metadata: bool = True, **kwargs) -> None:
        """Export results to CSV file.
        
        Args:
            file_path: Path to save CSV file
            flatten_metadata: Whether to flatten metadata into columns
            **kwargs: Additional pandas to_csv options
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data = []
        
        for result in self.results:
            row = {}
            
            # Add metadata
            if flatten_metadata:
                for key, value in result.metadata.items():
                    row[f"meta_{key}"] = value
            
            # Add summary stats
            for key, value in result.summary_stats.items():
                row[f"stats_{key}"] = value
            
            # Add turn-by-turn data as aggregated metrics
            if result.conversation_log:
                dcs_scores = [turn.get('dcs_score', 0) for turn in result.conversation_log 
                            if isinstance(turn.get('dcs_score'), int)]
                hes_scores = [turn.get('hes_score', 0) for turn in result.conversation_log
                            if isinstance(turn.get('hes_score'), int)]
                sis_scores = [turn.get('sis_score', 0) for turn in result.conversation_log
                            if isinstance(turn.get('sis_score'), int)]
                
                if dcs_scores:
                    row['turn_max_dcs'] = max(dcs_scores)
                    row['turn_min_dcs'] = min(dcs_scores)
                if hes_scores:
                    row['turn_max_hes'] = max(hes_scores)
                    row['turn_min_hes'] = min(hes_scores)
                if sis_scores:
                    row['turn_sis_count'] = len([s for s in sis_scores if s > 0])
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, **kwargs)
        
        print(f"✅ Exported {len(self.results)} results to {file_path}")


class ExcelExporter(BaseExporter):
    """Export experiment results to Excel format with multiple sheets."""
    
    def export(self, file_path: Union[str, Path], include_raw_conversations: bool = False, **kwargs) -> None:
        """Export results to Excel file with multiple sheets.
        
        Args:
            file_path: Path to save Excel file
            include_raw_conversations: Whether to include raw conversation data
            **kwargs: Additional pandas ExcelWriter options
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(file_path, engine='openpyxl', **kwargs) as writer:
            # Summary sheet
            summary_data = []
            for result in self.results:
                row = {
                    'Model': result.metadata['target_model'],
                    'Case': result.metadata['case_name'],
                    'Theme': result.metadata.get('theme', 'Unknown'),
                    'Condition': result.metadata.get('condition', 'Unknown'),
                    'Harm_Type': result.metadata.get('harm_type', 'Unknown'),
                    'Total_Turns': result.summary_stats['total_turns'],
                    'Avg_DCS': result.summary_stats['avg_dcs'],
                    'Avg_HES': result.summary_stats['avg_hes'],
                    'Total_SIS': result.summary_stats['total_sis'],
                    'Max_DCS': result.summary_stats['max_dcs'],
                    'Max_HES': result.summary_stats['max_hes'],
                    'Timestamp': result.metadata.get('timestamp', 'Unknown')
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Model comparison sheet
            model_stats = {}
            for result in self.results:
                model = result.metadata['target_model']
                if model not in model_stats:
                    model_stats[model] = {
                        'Experiments': 0,
                        'DCS_Scores': [],
                        'HES_Scores': [],
                        'SIS_Totals': []
                    }
                
                model_stats[model]['Experiments'] += 1
                model_stats[model]['DCS_Scores'].append(result.summary_stats['avg_dcs'])
                model_stats[model]['HES_Scores'].append(result.summary_stats['avg_hes'])
                model_stats[model]['SIS_Totals'].append(result.summary_stats['total_sis'])
            
            model_comparison = []
            for model, stats in model_stats.items():
                row = {
                    'Model': model,
                    'Experiments': stats['Experiments'],
                    'Avg_DCS': sum(stats['DCS_Scores']) / len(stats['DCS_Scores']),
                    'Avg_HES': sum(stats['HES_Scores']) / len(stats['HES_Scores']),
                    'Total_SIS': sum(stats['SIS_Totals']),
                    'Max_DCS': max(stats['DCS_Scores']),
                    'Max_HES': max(stats['HES_Scores'])
                }
                model_comparison.append(row)
            
            model_df = pd.DataFrame(model_comparison)
            model_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Theme analysis sheet
            theme_stats = {}
            for result in self.results:
                theme = result.metadata.get('theme', 'Unknown')
                if theme not in theme_stats:
                    theme_stats[theme] = {
                        'Experiments': 0,
                        'DCS_Scores': [],
                        'HES_Scores': [],
                        'SIS_Totals': []
                    }
                
                theme_stats[theme]['Experiments'] += 1
                theme_stats[theme]['DCS_Scores'].append(result.summary_stats['avg_dcs'])
                theme_stats[theme]['HES_Scores'].append(result.summary_stats['avg_hes'])
                theme_stats[theme]['SIS_Totals'].append(result.summary_stats['total_sis'])
            
            theme_analysis = []
            for theme, stats in theme_stats.items():
                row = {
                    'Theme': theme,
                    'Experiments': stats['Experiments'],
                    'Avg_DCS': sum(stats['DCS_Scores']) / len(stats['DCS_Scores']),
                    'Avg_HES': sum(stats['HES_Scores']) / len(stats['HES_Scores']),
                    'Total_SIS': sum(stats['SIS_Totals'])
                }
                theme_analysis.append(row)
            
            theme_df = pd.DataFrame(theme_analysis)
            theme_df.to_excel(writer, sheet_name='Theme_Analysis', index=False)
            
            # Raw conversations (if requested)
            if include_raw_conversations:
                conversation_data = []
                for result in self.results:
                    for turn in result.conversation_log:
                        row = {
                            'Model': result.metadata['target_model'],
                            'Case': result.metadata['case_name'],
                            'Turn_Number': turn['turn_number'],
                            'User_Message': turn['user'],
                            'Bot_Response': turn['bot'],
                            'DCS_Score': turn.get('dcs_score', 'N/A'),
                            'HES_Score': turn.get('hes_score', 'N/A'),
                            'SIS_Score': turn.get('sis_score', 'N/A')
                        }
                        conversation_data.append(row)
                
                if conversation_data:
                    conv_df = pd.DataFrame(conversation_data)
                    conv_df.to_excel(writer, sheet_name='Raw_Conversations', index=False)
        
        print(f"✅ Exported {len(self.results)} results to {file_path}")


def export_results(
    results: List[ExperimentResult],
    file_path: Union[str, Path],
    format: str = 'json',
    **kwargs
) -> None:
    """Convenience function to export results in various formats.
    
    Args:
        results: List of experiment results
        file_path: Path to save file
        format: Export format ('json', 'csv', 'excel')
        **kwargs: Format-specific options
    """
    exporters = {
        'json': JSONExporter,
        'csv': CSVExporter, 
        'excel': ExcelExporter
    }
    
    if format.lower() not in exporters:
        raise ValueError(f"Unsupported format '{format}'. Available: {list(exporters.keys())}")
    
    exporter = exporters[format.lower()](results)
    exporter.export(file_path, **kwargs)