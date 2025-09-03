"""Comprehensive conversation logging for psy-bench experiments."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import ExperimentResult, ScoreType, Turn


class ConversationLogger:
    """Logger for capturing detailed conversation logs and scores."""
    
    def __init__(self, output_dir: Union[str, Path] = "conversation_logs"):
        """Initialize the conversation logger.
        
        Args:
            output_dir: Directory to save logs (default: conversation_logs)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def log_experiment(
        self,
        result: ExperimentResult,
        format_type: str = "both"  # "json", "markdown", or "both"
    ) -> Dict[str, Path]:
        """Log a single experiment result.
        
        Args:
            result: The experiment result to log
            format_type: Output format ("json", "markdown", or "both")
            
        Returns:
            Dictionary with paths to created log files
        """
        # Generate base filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_case_name = self._safe_filename(result.test_case.name)
        safe_model_name = self._safe_filename(result.model)
        
        base_filename = f"{timestamp}_{safe_case_name}_{safe_model_name}"
        
        created_files = {}
        
        if format_type in ["json", "both"]:
            json_path = self.output_dir / f"{base_filename}.json"
            self._write_json_log(result, json_path)
            created_files["json"] = json_path
        
        if format_type in ["markdown", "both"]:
            md_path = self.output_dir / f"{base_filename}.md"
            self._write_markdown_log(result, md_path)
            created_files["markdown"] = md_path
        
        return created_files
    
    def log_batch_experiments(
        self,
        results: List[ExperimentResult],
        batch_name: str = "batch_experiment",
        format_type: str = "both"
    ) -> Dict[str, Path]:
        """Log multiple experiment results in a batch.
        
        Args:
            results: List of experiment results to log
            batch_name: Name for the batch (used in filename)
            format_type: Output format ("json", "markdown", or "both")
            
        Returns:
            Dictionary with paths to created log files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_batch_name = self._safe_filename(batch_name)
        
        base_filename = f"{timestamp}_{safe_batch_name}"
        
        created_files = {}
        
        if format_type in ["json", "both"]:
            json_path = self.output_dir / f"{base_filename}.json"
            self._write_batch_json_log(results, json_path)
            created_files["json"] = json_path
        
        if format_type in ["markdown", "both"]:
            md_path = self.output_dir / f"{base_filename}.md"
            self._write_batch_markdown_log(results, md_path, batch_name)
            created_files["markdown"] = md_path
        
        return created_files
    
    def _write_json_log(self, result: ExperimentResult, filepath: Path):
        """Write experiment result to JSON format."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment": {
                "model": result.model,
                "test_case": {
                    "id": result.test_case.id,
                    "name": result.test_case.name,
                    "theme": result.test_case.theme,
                    "condition": result.test_case.condition,
                    "harm_type": result.test_case.harm_type,
                    "total_prompts": len(result.test_case.prompts)
                },
                "summary": result.summary,
                "metadata": result.metadata
            },
            "conversation": {
                "turns": [self._turn_to_dict(turn) for turn in result.turns],
                "total_turns": len(result.turns)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _write_markdown_log(self, result: ExperimentResult, filepath: Path):
        """Write experiment result to Markdown format."""
        content = self._generate_markdown_content(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _write_batch_json_log(self, results: List[ExperimentResult], filepath: Path):
        """Write multiple experiment results to JSON format."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "batch_summary": {
                "total_experiments": len(results),
                "models": list(set(r.model for r in results)),
                "test_cases": list(set(r.test_case.name for r in results)),
                "total_turns": sum(len(r.turns) for r in results)
            },
            "experiments": []
        }
        
        for result in results:
            experiment_data = {
                "model": result.model,
                "test_case": {
                    "id": result.test_case.id,
                    "name": result.test_case.name,
                    "theme": result.test_case.theme,
                    "condition": result.test_case.condition,
                    "harm_type": result.test_case.harm_type
                },
                "summary": result.summary,
                "metadata": result.metadata,
                "conversation": {
                    "turns": [self._turn_to_dict(turn) for turn in result.turns],
                    "total_turns": len(result.turns)
                }
            }
            log_data["experiments"].append(experiment_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _write_batch_markdown_log(self, results: List[ExperimentResult], filepath: Path, batch_name: str):
        """Write multiple experiment results to Markdown format."""
        content = []
        
        # Header
        content.append(f"# Batch Experiment Log: {batch_name}")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Batch summary
        models = list(set(r.model for r in results))
        test_cases = list(set(r.test_case.name for r in results))
        total_turns = sum(len(r.turns) for r in results)
        
        content.append("## Batch Summary")
        content.append(f"- **Total Experiments:** {len(results)}")
        content.append(f"- **Models:** {len(models)} ({', '.join(models)})")
        content.append(f"- **Test Cases:** {len(test_cases)}")
        content.append(f"- **Total Conversation Turns:** {total_turns}")
        content.append("")
        
        # Individual experiments
        for i, result in enumerate(results, 1):
            content.append(f"## Experiment {i}: {result.test_case.name} × {result.model}")
            content.append(self._generate_markdown_content(result, include_header=False))
            content.append("\n---\n")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
    
    def _generate_markdown_content(self, result: ExperimentResult, include_header: bool = True) -> str:
        """Generate markdown content for a single experiment."""
        content = []
        
        if include_header:
            content.append(f"# Conversation Log: {result.test_case.name}")
            content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append("")
        
        # Experiment details
        content.append("## Experiment Details")
        content.append(f"- **Model:** {result.model}")
        content.append(f"- **Test Case:** {result.test_case.name}")
        content.append(f"- **Theme:** {result.test_case.theme}")
        content.append(f"- **Condition:** {result.test_case.condition}")
        content.append(f"- **Harm Type:** {result.test_case.harm_type}")
        content.append("")
        
        # Summary scores
        summary = result.summary
        content.append("## Summary Scores")
        content.append(f"- **Total Turns:** {summary['total_turns']}")
        content.append(f"- **Average DCS (Delusion Confirmation):** {summary['avg_dcs']:.3f}")
        content.append(f"- **Average HES (Harm Enablement):** {summary['avg_hes']:.3f}")
        content.append(f"- **Total SIS (Safety Intervention):** {summary['total_sis']}")
        content.append(f"- **Max DCS:** {summary['max_dcs']}")
        content.append(f"- **Max HES:** {summary['max_hes']}")
        content.append("")
        
        # Conversation turns
        content.append("## Conversation")
        
        for turn in result.turns:
            content.append(f"### Turn {turn.number}")
            content.append("")
            
            # User message
            content.append("**👤 User:**")
            content.append(turn.user_message)
            content.append("")
            
            # Bot response
            content.append("**🤖 Assistant:**")
            bot_lines = turn.bot_response.split('\n')
            for line in bot_lines:
                content.append(line)
            content.append("")
            
            # Scores
            if turn.scores:
                content.append("**📊 Scores:**")
                for score_type, score in turn.scores.items():
                    if score is not None:
                        score_name = self._score_type_name(score_type)
                        content.append(f"- {score_name}: {score}/5")
                content.append("")
            
            content.append("---")
            content.append("")
        
        # Metadata
        if result.metadata:
            content.append("## Metadata")
            for key, value in result.metadata.items():
                if isinstance(value, dict):
                    content.append(f"- **{key}:**")
                    for sub_key, sub_value in value.items():
                        content.append(f"  - {sub_key}: {sub_value}")
                else:
                    content.append(f"- **{key}:** {value}")
            content.append("")
        
        return '\n'.join(content)
    
    def _turn_to_dict(self, turn: Turn) -> Dict:
        """Convert a Turn object to dictionary for JSON serialization."""
        return {
            "number": turn.number,
            "user_message": turn.user_message,
            "bot_response": turn.bot_response,
            "scores": {
                score_type.value: score 
                for score_type, score in turn.scores.items() 
                if score is not None
            },
            "metadata": turn.metadata
        }
    
    def _score_type_name(self, score_type: ScoreType) -> str:
        """Get human-readable name for score type."""
        names = {
            ScoreType.DCS: "DCS (Delusion Confirmation Score)",
            ScoreType.HES: "HES (Harm Enablement Score)",
            ScoreType.SIS: "SIS (Safety Intervention Score)"
        }
        return names.get(score_type, str(score_type))
    
    def _safe_filename(self, name: str) -> str:
        """Convert a name to a safe filename."""
        # Replace spaces and special characters with underscores
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
        # Remove multiple consecutive underscores
        while '__' in safe:
            safe = safe.replace('__', '_')
        # Remove leading/trailing underscores
        return safe.strip('_')