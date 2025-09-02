"""Batch processing for multiple experiments."""

import asyncio
from typing import List, Optional

from .models import ExperimentResult, RunConfig, TestCase
from .runner import ExperimentRunner


class BatchRunner:
    """Handles batch experiment execution."""
    
    def __init__(self, runner: ExperimentRunner):
        """Initialize batch runner.
        
        Args:
            runner: Single experiment runner
        """
        self.runner = runner
    
    def run_batch(
        self,
        test_cases: List[TestCase],
        models: List[str],
        config: Optional[RunConfig] = None
    ) -> List[ExperimentResult]:
        """Run experiments for multiple cases and models.
        
        Args:
            test_cases: Test cases to run
            models: Models to test
            config: Run configuration
            
        Returns:
            List of all experiment results
        """
        config = config or RunConfig()
        results = []
        total = len(test_cases) * len(models)
        current = 0
        
        print(f"\n🚀 Running batch: {len(models)} models × {len(test_cases)} cases = {total} experiments")
        
        for model in models:
            for test_case in test_cases:
                current += 1
                progress = (current / total) * 100
                print(f"\n[{current}/{total}] ({progress:.1f}%) ", end="")
                
                try:
                    result = self.runner.run(test_case, model, config)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Failed: {model} on {test_case.name} - {e}")
        
        print(f"\n✅ Batch complete: {len(results)}/{total} successful")
        return results
    
    async def run_batch_async(
        self,
        test_cases: List[TestCase],
        models: List[str],
        config: Optional[RunConfig] = None,
        max_concurrent: int = 10,
    ) -> List[ExperimentResult]:
        """Run experiments concurrently.
        
        Args:
            test_cases: Test cases to run
            models: Models to test
            config: Run configuration
            max_concurrent: Maximum concurrent experiments
            
        Returns:
            List of all experiment results
        """
        config = config or RunConfig()
        
        # Create all tasks
        tasks = []
        for model in models:
            for test_case in test_cases:
                task = self._run_async(test_case, model, config)
                tasks.append(task)
        
        total = len(tasks)
        print(f"\n🚀 Running async batch: {total} experiments (max {max_concurrent} concurrent)")
        
        # Run with concurrency limit and progress tracking
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        
        async def run_with_limit_and_progress(task):
            nonlocal completed
            async with semaphore:
                result = await task
                completed += 1
                # Simple progress indicator
                progress = (completed / total) * 100
                print(f"\r📊 Progress: {completed}/{total} ({progress:.1f}%) completed", end="", flush=True)
                return result
        
        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_limit_and_progress(task) for task in tasks],
            return_exceptions=True
        )
        
        # Clear the progress line
        print()  # New line after progress
        
        # Filter successful results
        successful = [r for r in results if isinstance(r, ExperimentResult)]
        failed = len(results) - len(successful)
        
        print(f"\n✅ Async batch complete: {len(successful)}/{total} successful")
        if failed > 0:
            print(f"   {failed} experiments failed")
        
        return successful
    
    async def _run_async(
        self,
        test_case: TestCase,
        model: str,
        config: RunConfig
    ) -> ExperimentResult:
        """Run a single experiment asynchronously."""
        # For async, we need to create a new runner with async operations
        # This is simplified for now - in production would use async runner
        return await asyncio.to_thread(
            self.runner.run,
            test_case,
            model,
            config
        )
