#!/usr/bin/env python3
"""Example: Run all test cases against multiple models."""

from psy_bench import Experiment
from psy_bench.cases import ARCHETYPAL_CASES, CASE_NAMES
from psy_bench.analysis import ExperimentReporter, export_results

def main():
    """Run all archetypal cases against multiple models."""
    
    # Models to test
    models_to_test = [
        "google/gemini-2.5-flash",
        "openai/gpt-4o-mini",
        # Add more models as needed
    ]
    
    print(f"🧪 Running comprehensive evaluation:")
    print(f"   Models: {len(models_to_test)}")
    print(f"   Cases: {len(ARCHETYPAL_CASES)} (all archetypal cases)")
    print(f"   Total experiments: {len(models_to_test)} × {len(ARCHETYPAL_CASES)} = {len(models_to_test) * len(ARCHETYPAL_CASES)}")
    print()
    
    # Create experiment
    experiment = Experiment()
    
    # Option 1: Use all case names
    all_case_names = CASE_NAMES
    
    # Option 2: Use all case objects directly  
    # all_cases = ARCHETYPAL_CASES
    
    try:
        # Run batch experiment
        print("🚀 Starting batch experiment...")
        results = experiment.run_batch(
            cases=all_case_names,  # or all_cases
            models=models_to_test,
            verbose=True
        )
        
        print("\\n" + "="*80)
        print("COMPREHENSIVE EVALUATION COMPLETE")
        print("="*80)
        
        # Generate comprehensive report
        reporter = ExperimentReporter(results)
        summary = reporter.generate_summary_report(verbose=True)
        
        # Export results in multiple formats
        print("\\n📊 Exporting results...")
        export_results(results, "comprehensive_results.json", format="json")
        export_results(results, "comprehensive_results.xlsx", format="excel", 
                      include_raw_conversations=True)
        
        print("✅ Comprehensive evaluation completed successfully!")
        print(f"   Results exported to comprehensive_results.json and .xlsx")
        
    except Exception as e:
        print(f"❌ Batch experiment failed: {e}")


def run_subset_demo():
    """Demo with a smaller subset for testing."""
    print("\\n" + "="*60)
    print("SUBSET DEMO (First 4 cases)")
    print("="*60)
    
    experiment = Experiment()
    
    # Just test first 4 cases for demo
    subset_cases = CASE_NAMES[:4]
    models = ["google/gemini-2.5-flash"]
    
    results = experiment.run_batch(
        cases=subset_cases,
        models=models,
        verbose=True
    )
    
    # Quick summary
    reporter = ExperimentReporter(results)
    df = reporter.to_dataframe()
    
    print("\\n📈 Quick Results Summary:")
    print(df[['case', 'avg_dcs', 'avg_hes', 'total_sis']].to_string(index=False))


if __name__ == "__main__":
    # Comment out main() and run subset_demo() for testing
    # main()
    run_subset_demo()