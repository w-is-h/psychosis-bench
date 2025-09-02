#!/usr/bin/env python3
"""Demo: Show how to run all cases against multiple models (without actually running)."""

from psy_bench import Experiment
from psy_bench.cases import ARCHETYPAL_CASES, CASE_NAMES

def main():
    """Demo the API for running all cases against multiple models."""
    
    # Models to test
    models_to_test = [
        "google/gemini-2.5-flash",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20241022"
    ]
    
    print("🧪 Psy-Bench: Run All Cases Demo")
    print("=" * 50)
    
    print(f"📊 Available options:")
    print(f"   Total archetypal cases: {len(ARCHETYPAL_CASES)}")
    print(f"   Models to test: {len(models_to_test)}")
    print(f"   Total experiments: {len(models_to_test)} × {len(ARCHETYPAL_CASES)} = {len(models_to_test) * len(ARCHETYPAL_CASES)}")
    print()
    
    print("🎭 Available test cases:")
    for i, case in enumerate(ARCHETYPAL_CASES[:4], 1):  # Show first 4
        print(f"   {i}. {case.name}")
        print(f"      Theme: {case.theme}")
        print(f"      Harm: {case.harm_type}")
    print(f"   ... and {len(ARCHETYPAL_CASES) - 4} more cases")
    print()
    
    # Show the different ways to run experiments
    print("🚀 Usage Options:")
    print()
    
    print("1. Run ALL cases against multiple models:")
    print("   ```python")
    print("   experiment = Experiment()")
    print("   results = experiment.run_all_cases(models=[")
    for model in models_to_test:
        print(f"       '{model}',")
    print("   ])")
    print("   ```")
    print()
    
    print("2. Run specific cases against multiple models:")
    print("   ```python")
    print("   results = experiment.run_batch(")
    print("       cases=['Case 1.1: The Conduit [EXPLICIT]', 'Case 2.1: The Digital Romance [EXPLICIT]'],")
    print("       models=['google/gemini-2.5-flash', 'openai/gpt-4o-mini']")
    print("   )")
    print("   ```")
    print()
    
    print("3. Use case objects directly:")
    print("   ```python")
    print("   from psy_bench.cases import ARCHETYPAL_CASES")
    print("   results = experiment.run_batch(")
    print("       cases=ARCHETYPAL_CASES[:4],  # First 4 cases")
    print("       models=models_to_test")
    print("   )")
    print("   ```")
    print()
    
    print("4. Filter cases by type:")
    print("   ```python")
    print("   from psy_bench.cases.archetypal import get_explicit_cases, get_implicit_cases")
    print("   explicit_cases = get_explicit_cases()  # Only explicit versions")
    print("   results = experiment.run_batch(cases=explicit_cases, models=models_to_test)")
    print("   ```")
    print()
    
    print("📈 Analysis and Export:")
    print("   ```python")
    print("   from psy_bench.analysis import ExperimentReporter, export_results")
    print("   ")
    print("   reporter = ExperimentReporter(results)")
    print("   summary = reporter.generate_summary_report()")
    print("   ")
    print("   export_results(results, 'results.json', format='json')")
    print("   export_results(results, 'results.xlsx', format='excel')")
    print("   ```")
    print()
    
    print("💡 Pro tip: Start with a subset of cases for testing!")
    print("   For example, test with just 2-3 cases first to verify everything works.")


if __name__ == "__main__":
    main()