#!/usr/bin/env python3
"""Quick start example for psy-bench."""

from psy_bench import PsyBench

def main():
    """Run a simple experiment."""
    # Initialize PsyBench
    # API key can be set via:
    # 1. Parameter: PsyBench(api_key="...")
    # 2. Environment: export OPENROUTER_API_KEY="..."
    # 3. .env file: OPENROUTER_API_KEY=...
    
    bench = PsyBench()
    
    # Run a single experiment
    print("Running single experiment...")
    result = bench.run(
        case="Case 1.1: The Conduit [EXPLICIT]",
        model="google/gemini-2.5-flash",  # Optional, uses default if not specified
        start_turn=1,
        end_turn=10,  # Just first 10 turns for quick demo
    )
    
    # Show summary
    print(f"\nExperiment completed!")
    print(f"Model: {result.model}")
    print(f"Case: {result.test_case.name}")
    print(f"Summary: {result.summary}")
    
    # Run a small batch
    print("\n" + "="*60)
    print("Running batch experiment...")
    
    results = bench.run_batch(
        cases=[
            "Case 1.1: The Conduit [EXPLICIT]",
            "Case 2.1: The Digital Romance [EXPLICIT]"
        ],
        models=["google/gemini-2.5-flash", "openai/gpt-4"],
        verbose=False  # Less output
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    analyzer = bench.analyze(results)
    
    # Export results
    analyzer.export("json", "quickstart_results.json")
    print("\nResults exported to quickstart_results.json")
    
    # List available cases
    print("\n" + "="*60)
    print("Available test cases:")
    for case_name in bench.list_cases()[:5]:  # Show first 5
        print(f"  - {case_name}")
    print(f"  ... and {len(bench.list_cases()) - 5} more")
    
    # Show themes
    print(f"\nAvailable themes: {', '.join(bench.get_themes())}")


if __name__ == "__main__":
    main()
