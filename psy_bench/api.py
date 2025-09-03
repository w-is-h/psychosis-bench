"""Main API entry point for psy-bench."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union

from .analysis import ExportFormat, ResultAnalyzer
from .config import DEFAULT_JUDGE_MODEL, DEFAULT_TARGET_MODEL, OPENROUTER_API_KEY
from .core.batch import BatchRunner
from .core.cases import CaseLoader
from .core.client import OpenRouterClient
from .core.conversation_logger import ConversationLogger
from .core.models import ExperimentResult, RunConfig, TestCase
from .core.runner import ExperimentRunner
from .core.scoring import Scorer


class PsyBench:
    """Main entry point for the psy-bench library."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        sis_use_llm: bool = True,
        enable_conversation_logging: bool = True,
        log_output_dir: Union[str, Path] = "conversation_logs",
        **config
    ):
        """Initialize PsyBench.
        
        Args:
            api_key: OpenRouter API key (uses settings if not provided)
            default_model: Default model for experiments
            judge_model: Model to use for scoring
            sis_use_llm: Whether to use LLM for SIS scoring
            enable_conversation_logging: Enable automatic conversation logging
            log_output_dir: Directory for conversation logs
            **config: Additional configuration options
        """
        # Configuration
        self.api_key = api_key or OPENROUTER_API_KEY
        self.default_model = default_model or DEFAULT_TARGET_MODEL
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL
        self.sis_use_llm = sis_use_llm
        self.enable_conversation_logging = enable_conversation_logging
        
        # Avoid printing secrets (API key) on initialization
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set it via api_key parameter or OPENROUTER_API_KEY environment variable."
            )
        
        # Initialize components
        self._client = OpenRouterClient(api_key=self.api_key, **config)
        self._scorer = Scorer(self._client, self.judge_model, sis_use_llm=self.sis_use_llm)
        self._runner = ExperimentRunner(self._client, self._scorer)
        self._batch_runner = BatchRunner(self._runner)
        
        # Initialize conversation logger
        if self.enable_conversation_logging:
            self._conversation_logger = ConversationLogger(log_output_dir)
        else:
            self._conversation_logger = None
        
        # Load bundled test cases
        self._cases = CaseLoader.load_bundled_cases()
    
    def run(
        self,
        case: Union[str, TestCase],
        model: Optional[str] = None,
        start_turn: int = 1,
        end_turn: int = 12,
        verbose: bool = True
    ) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            case: Test case name or TestCase object
            model: Model to test (uses default if not provided)
            start_turn: Starting turn (1-12)
            end_turn: Ending turn (1-12)
            verbose: Whether to print progress
            
        Returns:
            ExperimentResult with conversation and scores
            
        Example:
            >>> bench = PsyBench()
            >>> result = bench.run("Case 1.1: The Conduit [EXPLICIT]")
            >>> print(result.summary)
        """
        # Resolve test case
        if isinstance(case, str):
            test_case = CaseLoader.get_case_by_name(case, self._cases)
            if not test_case:
                raise ValueError(f"Test case '{case}' not found")
        else:
            test_case = case
        
        # Create config
        config = RunConfig(
            start_turn=start_turn,
            end_turn=end_turn,
            verbose=verbose,
            judge_model=self.judge_model
        )
        
        # Run experiment
        result = self._runner.run(
            test_case=test_case,
            model=model or self.default_model,
            config=config
        )
        
        # Log conversation if enabled
        if self._conversation_logger:
            log_files = self._conversation_logger.log_experiment(result)
            if verbose:
                for format_type, path in log_files.items():
                    print(f"📝 Conversation log ({format_type}): {path}")
        
        return result
    
    def run_batch(
        self,
        cases: Optional[List[Union[str, TestCase]]] = None,
        models: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Run multiple experiments.
        
        Args:
            cases: List of test cases (uses all if not provided)
            models: List of models (uses default if not provided)
            verbose: Whether to print progress
            
        Returns:
            List of experiment results
            
        Example:
            >>> bench = PsyBench()
            >>> results = bench.run_batch(
            ...     cases=["Case 1.1: The Conduit [EXPLICIT]", 
            ...            "Case 2.1: The Digital Romance [EXPLICIT]"],
            ...     models=["gpt-4", "claude-3"]
            ... )
        """
        # Default to all cases if not specified
        if cases is None:
            test_cases = self._cases
        else:
            test_cases = []
            for case in cases:
                if isinstance(case, str):
                    tc = CaseLoader.get_case_by_name(case, self._cases)
                    if tc:
                        test_cases.append(tc)
                else:
                    test_cases.append(case)
        
        # Default to configured model if not specified
        if models is None:
            models = [self.default_model]
        
        # Run batch
        config = RunConfig(verbose=verbose, judge_model=self.judge_model)
        results = self._batch_runner.run_batch(test_cases, models, config)
        
        # Log batch conversation if enabled
        if self._conversation_logger and results:
            batch_name = f"batch_{len(models)}models_{len(test_cases)}cases"
            log_files = self._conversation_logger.log_batch_experiments(results, batch_name)
            if verbose:
                for format_type, path in log_files.items():
                    print(f"📝 Batch conversation log ({format_type}): {path}")
        
        return results
    
    async def run_batch_async(
        self,
        cases: Optional[List[Union[str, TestCase]]] = None,
        models: Optional[List[str]] = None,
        max_concurrent: int = 10,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Run multiple experiments concurrently.
        
        Args:
            cases: List of test cases (uses all if not provided)
            models: List of models (uses default if not provided)
            max_concurrent: Maximum concurrent experiments
            verbose: Whether to print progress
            
        Returns:
            List of experiment results
            
        Example:
            >>> bench = PsyBench()
            >>> results = await bench.run_batch_async(
            ...     models=["gpt-4", "claude-3", "gemini-pro"],
            ...     max_concurrent=20
            ... )
        """
        # Default to all cases if not specified
        if cases is None:
            test_cases = self._cases
        else:
            test_cases = []
            for case in cases:
                if isinstance(case, str):
                    tc = CaseLoader.get_case_by_name(case, self._cases)
                    if tc:
                        test_cases.append(tc)
                else:
                    test_cases.append(case)
        
        # Default to configured model if not specified
        if models is None:
            models = [self.default_model]
        
        # Run async batch
        config = RunConfig(verbose=verbose, judge_model=self.judge_model)
        results = await self._batch_runner.run_batch_async(
            test_cases, models, config, max_concurrent
        )
        
        # Log batch conversation if enabled
        if self._conversation_logger and results:
            batch_name = f"async_batch_{len(models)}models_{len(test_cases)}cases"
            log_files = self._conversation_logger.log_batch_experiments(results, batch_name)
            if verbose:
                for format_type, path in log_files.items():
                    print(f"📝 Async batch conversation log ({format_type}): {path}")
        
        return results
    
    def analyze(
        self,
        results: List[ExperimentResult],
        print_summary: bool = True
    ) -> ResultAnalyzer:
        """Analyze experiment results.
        
        Args:
            results: List of experiment results
            print_summary: Whether to print summary
            
        Returns:
            ResultAnalyzer instance for further analysis
            
        Example:
            >>> results = bench.run_batch()
            >>> analyzer = bench.analyze(results)
            >>> analyzer.export(ExportFormat.EXCEL, "results.xlsx")
        """
        analyzer = ResultAnalyzer(results)
        
        if print_summary:
            analyzer.print_summary()
        
        return analyzer
    
    def list_cases(self, theme: Optional[str] = None) -> List[str]:
        """List available test cases.
        
        Args:
            theme: Filter by theme (shows all if not provided)
            
        Returns:
            List of test case names
        """
        if theme:
            cases = CaseLoader.get_cases_by_theme(theme, self._cases)
        else:
            cases = self._cases
        
        return [case.name for case in cases]
    
    def get_themes(self) -> List[str]:
        """Get list of available themes.
        
        Returns:
            List of unique themes
        """
        return list(set(case.theme for case in self._cases))
    
    def get_case_theme(self, case_name: str) -> Optional[str]:
        """Get the theme for a specific case.
        
        Args:
            case_name: Name of the test case
            
        Returns:
            Theme name or None if case not found
        """
        case = CaseLoader.get_case_by_name(case_name, self._cases)
        return case.theme if case else None
    
    def load_cases_from_file(self, file_path: Union[str, Path]) -> List[TestCase]:
        """Load custom test cases from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of loaded test cases
        """
        return CaseLoader.load_from_file(file_path)
    
    def log_conversations(
        self,
        results: Union[ExperimentResult, List[ExperimentResult]],
        format_type: str = "both",
        batch_name: Optional[str] = None
    ) -> Dict[str, Path]:
        """Manually log conversations for existing experiment results.
        
        Args:
            results: Single result or list of results to log
            format_type: Output format ("json", "markdown", or "both")
            batch_name: Name for batch logging (auto-generated if not provided)
            
        Returns:
            Dictionary with paths to created log files
            
        Example:
            >>> results = bench.run_batch()
            >>> log_files = bench.log_conversations(results, format_type="markdown")
        """
        if not self._conversation_logger:
            raise ValueError("Conversation logging is disabled. Enable it in PsyBench initialization.")
        
        if isinstance(results, ExperimentResult):
            # Single experiment
            return self._conversation_logger.log_experiment(results, format_type)
        else:
            # Batch of experiments
            if not batch_name:
                batch_name = f"manual_batch_{len(results)}experiments"
            return self._conversation_logger.log_batch_experiments(results, batch_name, format_type)