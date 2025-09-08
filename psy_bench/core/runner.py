"""Core experiment runner for psy-bench."""

import time
from typing import List, Optional

from .client import ChatMessage, OpenRouterClient
from .models import ExperimentResult, RunConfig, ScoreType, TestCase, Turn
from .scoring import Scorer


class ExperimentRunner:
    """Runs single experiments with AI models."""
    
    def __init__(self, client: OpenRouterClient, scorer: Scorer):
        """Initialize the runner.
        
        Args:
            client: API client for model interactions
            scorer: Scorer for evaluating responses
        """
        self.client = client
        self.scorer = scorer
    
    def run(
        self,
        test_case: TestCase,
        model: str,
        config: Optional[RunConfig] = None
    ) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            test_case: Test case to run
            model: Model to test
            config: Run configuration
            
        Returns:
            ExperimentResult with all data
        """
        config = config or RunConfig()
        
        # Select prompts based on turn range
        prompts = test_case.prompts[config.start_turn - 1:config.end_turn]
        
        if config.verbose:
            print(f"\n{'='*60}")
            print(f"Running: {model} on {test_case.name}")
            print(f"Turns: {config.start_turn}-{config.end_turn}")
            print(f"{'='*60}")
        
        # Initialize conversation
        messages: List[ChatMessage] = []
        turns: List[Turn] = []
        
        for i, prompt in enumerate(prompts):
            turn_number = config.start_turn + i
            
            if config.verbose:
                print(f"\n👤 Turn {turn_number}: {prompt[:80]}...")
            
            # Get model response
            messages.append(ChatMessage(role="user", content=prompt))
            
            try:
                response = self.client.chat(model, messages)
                bot_response = response.content
                messages.append(ChatMessage(role="assistant", content=bot_response))
                
                if config.verbose:
                    print(f"🤖 Response: {bot_response[:80]}...")
                
            except Exception as e:
                bot_response = f"ERROR: {e}"
                if config.verbose:
                    print(f"❌ Error: {e}")
            
            # Create turn
            turn = Turn(
                number=turn_number,
                user_message=prompt,
                bot_response=bot_response
            )
            turns.append(turn)
            
            # Score this turn
            scores = self.scorer.score_all_metrics(turns, test_case.harm_type, current_turn_number = turn_number)
            
            # Add valid scores to turn
            for score_type, result in scores.items():
                if result.is_valid:
                    turn.scores[score_type] = result.score
            
            if config.verbose:
                score_str = ", ".join([
                    f"{st.value.upper()}={turn.scores.get(st, 'N/A')}"
                    for st in ScoreType
                ])
                print(f"📊 Scores: {score_str}")
            
            # Delay between turns
            if config.delay_between_turns > 0 and i < len(prompts) - 1:
                time.sleep(config.delay_between_turns)
        
        # Create result
        result = ExperimentResult(
            test_case=test_case,
            model=model,
            turns=turns,
            metadata={
                "config": config.model_dump(),
                "turns_run": f"{config.start_turn}-{config.end_turn}",
                "judge_model": self.scorer.judge_model
            }
        )
        
        if config.verbose:
            summary = result.summary
            print(f"\n{'='*60}")
            print(f"Summary: Avg DCS={summary['avg_dcs']:.2f}, "
                  f"Avg HES={summary['avg_hes']:.2f}, "
                  f"Total SIS={summary['total_sis']}")
            print(f"{'='*60}")
        
        return result
