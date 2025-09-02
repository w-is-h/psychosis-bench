"""Main experiment runner for psy-bench."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..config.settings import settings
from ..cases.archetypal import TestCase, get_case_by_name
from .api_client import OpenRouterClient, ChatMessage
from .scoring import (
    ConversationTurn, 
    DelusionConfirmationScorer,
    HarmEnablementScorer, 
    SafetyInterventionScorer,
    ScoringResult
)


class ExperimentResult(BaseModel):
    """Results from running an experiment."""
    metadata: Dict[str, Any]
    conversation_log: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""
    target_model: str
    judge_model: str
    start_turn: int = 1
    end_turn: int = 12
    api_key: Optional[str] = None
    delay_between_turns: float = 1.0


class Experiment:
    """Main experiment runner for AI safety evaluation."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize the experiment.
        
        Args:
            config: Optional configuration. If None, uses default settings.
        """
        self.config = config or ExperimentConfig(
            target_model=settings.default_target_model,
            judge_model=settings.default_judge_model,
            api_key=settings.openrouter_api_key
        )
        
        # Validate setup
        is_valid, errors = settings.validate_setup()
        if not is_valid:
            raise ValueError(f"Setup validation failed: {'; '.join(errors)}")
        
        # Initialize API client
        self.client = OpenRouterClient(self.config.api_key)
        
        # Initialize scorers
        self.dcs_scorer = DelusionConfirmationScorer(self.client, self.config.judge_model)
        self.hes_scorer = HarmEnablementScorer(self.client, self.config.judge_model)
        self.sis_scorer = SafetyInterventionScorer(self.client, self.config.judge_model)
    
    def run_case(
        self, 
        case: Union[str, TestCase], 
        start_turn: Optional[int] = None,
        end_turn: Optional[int] = None,
        verbose: bool = True
    ) -> ExperimentResult:
        """Run an experiment on a single test case.
        
        Args:
            case: Test case name or TestCase object
            start_turn: Starting turn number (1-12)
            end_turn: Ending turn number (1-12)  
            verbose: Whether to print progress updates
            
        Returns:
            ExperimentResult with conversation log and analysis
        """
        # Get test case
        if isinstance(case, str):
            test_case = get_case_by_name(case)
            if not test_case:
                raise ValueError(f"Test case '{case}' not found")
        else:
            test_case = case
        
        # Determine turn range
        start_idx = (start_turn or self.config.start_turn) - 1
        end_idx = end_turn or self.config.end_turn
        
        prompts_to_run = test_case.prompts[start_idx:end_idx]
        
        if not prompts_to_run:
            raise ValueError(f"No prompts in range {start_idx+1}-{end_idx}")
        
        if verbose:
            print(f"--- Starting Experiment: {self.config.target_model} on '{test_case.name}' ---")
            print(f"Running turns {start_idx+1}-{end_idx} ({len(prompts_to_run)} prompts)")
        
        # Run the conversation
        conversation_log = []
        messages = []
        
        for i, prompt in enumerate(prompts_to_run):
            turn_number = start_idx + i + 1
            
            if verbose:
                print(f"\\n👤 USER (Turn {turn_number}): {prompt}")
            
            # Send message to target model
            messages.append(ChatMessage(role="user", content=prompt))
            
            try:
                if verbose:
                    print(f"   Sending to {self.config.target_model}...")
                
                response = self.client.chat(self.config.target_model, messages)
                bot_response = response.content
                messages.append(ChatMessage(role="assistant", content=bot_response))
                
                if verbose:
                    print(f"   ...Model responded.")
                    print(f"🤖 BOT (Turn {turn_number}):")
                    print(f"   {bot_response[:200]}{'...' if len(bot_response) > 200 else ''}")
                
            except Exception as e:
                bot_response = f"ERROR: {e}"
                messages.append(ChatMessage(role="assistant", content="Error in response."))
                if verbose:
                    print(f"   ❌ Error: {e}")
            
            # Create conversation turn
            turn = ConversationTurn(
                user=prompt,
                bot=bot_response,
                turn_number=turn_number
            )
            
            # Score this turn
            turn_scores = self._score_turn(
                conversation_log + [turn], 
                turn_number, 
                test_case.harm_type,
                verbose
            )
            
            # Create turn data
            turn_data = {
                "user": prompt,
                "bot": bot_response,
                "turn_number": turn_number,
                **turn_scores
            }
            
            conversation_log.append(turn_data)
            
            # Brief delay between turns
            if self.config.delay_between_turns > 0:
                time.sleep(self.config.delay_between_turns)
        
        if verbose:
            print("\\n--- Experiment Complete ---")
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(conversation_log)
        
        # Create result
        result = ExperimentResult(
            metadata={
                "target_model": self.config.target_model,
                "judge_model": self.config.judge_model,
                "case_name": test_case.name,
                "theme": test_case.theme,
                "condition": test_case.condition,
                "harm_type": test_case.harm_type,
                "turns_run": f"{start_idx+1}-{end_idx}",
                "total_turns": len(conversation_log),
                "timestamp": datetime.now().isoformat()
            },
            conversation_log=conversation_log,
            summary_stats=summary_stats
        )
        
        return result
    
    def _score_turn(
        self, 
        conversation_history: List[ConversationTurn], 
        turn_number: int,
        harm_type: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Score a single turn using all applicable scorers.
        
        Args:
            conversation_history: Full conversation history including current turn
            turn_number: Current turn number
            harm_type: Type of harm for this test case
            verbose: Whether to print scoring progress
            
        Returns:
            Dictionary with scoring results
        """
        scores = {}
        
        if verbose:
            print(f"   --- Scoring turn {turn_number} ---")
        
        # Delusion Confirmation Score (starts at turn 4)
        if turn_number >= settings.dcs_start_turn:
            dcs_result = self.dcs_scorer.score(conversation_history)
            scores["dcs_score"] = dcs_result.score
            scores["dcs_raw"] = dcs_result.raw_response
            scores["dcs_error"] = dcs_result.error
        else:
            scores["dcs_score"] = "N/A"
            scores["dcs_raw"] = None
            scores["dcs_error"] = None
        
        # Harm Enablement Score (starts at turn 7)
        if turn_number >= settings.hes_start_turn:
            hes_result = self.hes_scorer.score(conversation_history, harm_type=harm_type)
            scores["hes_score"] = hes_result.score
            scores["hes_raw"] = hes_result.raw_response
            scores["hes_error"] = hes_result.error
        else:
            scores["hes_score"] = "N/A"
            scores["hes_raw"] = None
            scores["hes_error"] = None
        
        # Safety Intervention Score (starts at turn 7)
        if turn_number >= settings.hes_start_turn:
            sis_result = self.sis_scorer.score(conversation_history)
            scores["sis_score"] = sis_result.score
            scores["sis_raw"] = sis_result.raw_response
            scores["sis_error"] = sis_result.error
        else:
            scores["sis_score"] = "N/A"
            scores["sis_raw"] = None
            scores["sis_error"] = None
        
        if verbose:
            dcs_display = scores["dcs_score"] if scores["dcs_score"] != "N/A" else "N/A"
            hes_display = scores["hes_score"] if scores["hes_score"] != "N/A" else "N/A"
            sis_display = scores["sis_score"] if scores["sis_score"] != "N/A" else "N/A"
            print(f"   Scores: DCS={dcs_display}, HES={hes_display}, SIS={sis_display}")
        
        return scores
    
    def _calculate_summary_stats(self, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from conversation log.
        
        Args:
            conversation_log: List of conversation turns with scores
            
        Returns:
            Dictionary with summary statistics
        """
        # Extract numeric scores
        dcs_scores = [
            turn["dcs_score"] for turn in conversation_log 
            if isinstance(turn.get("dcs_score"), int) and turn["dcs_score"] != -99
        ]
        
        hes_scores = [
            turn["hes_score"] for turn in conversation_log
            if isinstance(turn.get("hes_score"), int) and turn["hes_score"] != -99
        ]
        
        sis_scores = [
            turn["sis_score"] for turn in conversation_log
            if isinstance(turn.get("sis_score"), int)
        ]
        
        # Calculate averages and totals
        stats = {
            "total_turns": len(conversation_log),
            "dcs_turns_scored": len(dcs_scores),
            "hes_turns_scored": len(hes_scores),
            "sis_turns_scored": len(sis_scores),
            "avg_dcs": sum(dcs_scores) / len(dcs_scores) if dcs_scores else 0,
            "avg_hes": sum(hes_scores) / len(hes_scores) if hes_scores else 0,
            "total_sis": sum(sis_scores),
            "max_dcs": max(dcs_scores) if dcs_scores else 0,
            "max_hes": max(hes_scores) if hes_scores else 0
        }
        
        return stats
    
    def run_batch(
        self,
        cases: List[Union[str, TestCase]],
        models: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Run experiments on multiple cases and/or models.
        
        Args:
            cases: List of test case names or TestCase objects
            models: List of model names to test (uses config.target_model if None)
            verbose: Whether to print progress updates
            
        Returns:
            List of ExperimentResult objects
        """
        if models is None:
            models = [self.config.target_model]
        
        results = []
        total_experiments = len(cases) * len(models)
        current_exp = 0
        
        if verbose:
            print(f"🚀 Starting batch experiment: {len(models)} models × {len(cases)} cases = {total_experiments} total")
        
        for model in models:
            # Update config for this model
            original_model = self.config.target_model
            self.config.target_model = model
            
            for case in cases:
                current_exp += 1
                if verbose:
                    print(f"\\n{'='*80}")
                    print(f"Experiment {current_exp}/{total_experiments}: {model} on {case}")
                    print(f"{'='*80}")
                
                try:
                    result = self.run_case(case, verbose=verbose)
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"❌ Experiment failed: {e}")
                    continue
            
            # Restore original model
            self.config.target_model = original_model
        
        if verbose:
            print(f"\\n✅ Batch experiment complete: {len(results)} successful experiments")
        
        return results
    
    def run_all_cases(
        self,
        models: List[str],
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Convenience method to run all archetypal cases against multiple models.
        
        Args:
            models: List of model names to test
            verbose: Whether to print progress updates
            
        Returns:
            List of ExperimentResult objects
        """
        from ..cases.archetypal import ARCHETYPAL_CASES
        return self.run_batch(cases=ARCHETYPAL_CASES, models=models, verbose=verbose)
    
    async def run_batch_async(
        self,
        cases: List[Union[str, TestCase]],
        models: Optional[List[str]] = None,
        max_concurrent: Optional[int] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Run experiments on multiple cases and/or models concurrently.
        
        Args:
            cases: List of test case names or TestCase objects
            models: List of model names to test (uses config.target_model if None)
            max_concurrent: Maximum concurrent requests. If None, uses settings.max_concurrent_requests
            verbose: Whether to print progress updates
            
        Returns:
            List of ExperimentResult objects
        """
        from .async_client import AsyncOpenRouterClient
        
        if models is None:
            models = [self.config.target_model]
        
        # Create async client  
        async_client = AsyncOpenRouterClient(
            api_key=self.config.api_key,
            max_concurrent=max_concurrent  # Will use settings default if None
        )
        
        total_experiments = len(cases) * len(models)
        
        if verbose:
            print(f"🚀 Starting ASYNC batch experiment: {len(models)} models × {len(cases)} cases = {total_experiments} total")
            print(f"📊 Max concurrent requests: {async_client.max_concurrent}")
        
        # Prepare all experiment combinations
        experiment_configs = []
        for model in models:
            for case in cases:
                if isinstance(case, str):
                    test_case = get_case_by_name(case)
                    if not test_case:
                        continue
                else:
                    test_case = case
                
                experiment_configs.append({
                    'model': model,
                    'case': test_case,
                    'id': f"{model}_{test_case.name}"
                })
        
        if verbose:
            print(f"⏱️  Estimated time reduction: ~{len(models)}x faster than sequential")
        
        # Process experiments concurrently
        results = []
        
        # Process in batches to manage memory and rate limits
        batch_size = max_concurrent * 2  # Process 2x concurrent limit at a time
        
        for i in range(0, len(experiment_configs), batch_size):
            batch = experiment_configs[i:i + batch_size]
            
            if verbose:
                print(f"\\n📦 Processing batch {i//batch_size + 1}/{(len(experiment_configs)-1)//batch_size + 1} ({len(batch)} experiments)")
            
            # Run batch experiments concurrently
            batch_tasks = []
            for config in batch:
                task = self._run_single_experiment_async(
                    async_client, 
                    config['model'], 
                    config['case'],
                    verbose=verbose
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    if verbose:
                        print(f"❌ Experiment failed: {batch[j]['id']} - {result}")
                else:
                    results.append(result)
        
        if verbose:
            print(f"\\n✅ Async batch experiment complete: {len(results)} successful experiments")
        
        return results
    
    async def _run_single_experiment_async(
        self,
        async_client,
        model: str,
        test_case: TestCase,
        verbose: bool = False
    ) -> ExperimentResult:
        """Run a single experiment asynchronously."""
        import aiohttp
        
        if verbose:
            print(f"🔄 Starting: {model} on {test_case.name}")
        
        conversation_log = []
        messages = []
        
        async with aiohttp.ClientSession() as session:
            for i, prompt in enumerate(test_case.prompts):
                turn_number = i + 1
                
                # Send message to target model
                messages.append(ChatMessage(role="user", content=prompt))
                
                try:
                    response = await async_client.chat(session, model, messages)
                    bot_response = response.content
                    messages.append(ChatMessage(role="assistant", content=bot_response))
                    
                except Exception as e:
                    bot_response = f"ERROR: {e}"
                    messages.append(ChatMessage(role="assistant", content="Error in response."))
                
                # Create conversation turn (simplified scoring for async)
                turn_data = {
                    "user": prompt,
                    "bot": bot_response,
                    "turn_number": turn_number,
                    "dcs_score": "ASYNC_SKIP",  # Skip scoring in async mode for speed
                    "hes_score": "ASYNC_SKIP",
                    "sis_score": "ASYNC_SKIP"
                }
                
                conversation_log.append(turn_data)
                
                # Brief delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
        
        # Calculate summary statistics (simplified for async)
        summary_stats = {
            "total_turns": len(conversation_log),
            "dcs_turns_scored": 0,
            "hes_turns_scored": 0, 
            "sis_turns_scored": 0,
            "avg_dcs": 0,
            "avg_hes": 0,
            "total_sis": 0,
            "max_dcs": 0,
            "max_hes": 0
        }
        
        # Create result
        result = ExperimentResult(
            metadata={
                "target_model": model,
                "judge_model": "ASYNC_SKIP",
                "case_name": test_case.name,
                "theme": test_case.theme,
                "condition": test_case.condition,
                "harm_type": test_case.harm_type,
                "turns_run": f"1-{len(test_case.prompts)}",
                "total_turns": len(conversation_log),
                "timestamp": datetime.now().isoformat(),
                "async_mode": True
            },
            conversation_log=conversation_log,
            summary_stats=summary_stats
        )
        
        if verbose:
            print(f"✅ Completed: {model} on {test_case.name}")
        
        return result
    
    def run_all_cases_async(
        self,
        models: List[str],
        max_concurrent: Optional[int] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """Async convenience method to run all archetypal cases against multiple models.
        
        Args:
            models: List of model names to test
            max_concurrent: Maximum concurrent requests. If None, uses settings.max_concurrent_requests
            verbose: Whether to print progress updates
            
        Returns:
            List of ExperimentResult objects
        """
        from ..cases.archetypal import ARCHETYPAL_CASES
        return asyncio.run(
            self.run_batch_async(
                cases=ARCHETYPAL_CASES,
                models=models,
                max_concurrent=max_concurrent,
                verbose=verbose
            )
        )