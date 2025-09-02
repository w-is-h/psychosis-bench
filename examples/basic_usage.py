#!/usr/bin/env python3
"""Basic usage example for psy-bench library."""

from psy_bench import Experiment
from psy_bench.config import Settings

def main():
    """Run a basic experiment example."""
    # Check if setup is valid
    settings = Settings()
    is_valid, errors = settings.validate_setup()
    
    if not is_valid:
        print("❌ Setup validation failed:")
        for error in errors:
            print(f"   {error}")
        print("\\nPlease check your .env file and ensure OPENROUTER_API_KEY is set.")
        return
    
    print("✅ Setup validation passed!")
    print(f"Using target model: {settings.default_target_model}")
    print(f"Using judge model: {settings.default_judge_model}")
    
    # Create experiment
    experiment = Experiment()
    
    # Run a single test case
    print("\\n" + "="*60)
    print("Running single experiment...")
    print("="*60)
    
    try:
        result = experiment.run_case(
            case="Case 1.1: The Conduit [EXPLICIT]",
            start_turn=1,
            end_turn=6,  # Run just the first 2 phases for demo
            verbose=True
        )
        
        print("\\n--- EXPERIMENT RESULTS ---")
        print(f"Case: {result.metadata['case_name']}")
        print(f"Model: {result.metadata['target_model']}")
        print(f"Turns completed: {result.summary_stats['total_turns']}")
        print(f"Average DCS: {result.summary_stats['avg_dcs']:.2f}")
        print(f"Average HES: {result.summary_stats['avg_hes']:.2f}")
        print(f"Total Safety Interventions: {result.summary_stats['total_sis']}")
        
        # Show conversation summary
        print("\\n--- CONVERSATION SUMMARY ---")
        for turn in result.conversation_log:
            print(f"Turn {turn['turn_number']}:")
            print(f"  User: {turn['user'][:100]}...")
            print(f"  Bot:  {turn['bot'][:100]}...")
            dcs = turn['dcs_score'] if turn['dcs_score'] != 'N/A' else 'N/A'
            hes = turn['hes_score'] if turn['hes_score'] != 'N/A' else 'N/A'  
            sis = turn['sis_score'] if turn['sis_score'] != 'N/A' else 'N/A'
            print(f"  Scores: DCS={dcs}, HES={hes}, SIS={sis}")
            print()
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")


if __name__ == "__main__":
    main()