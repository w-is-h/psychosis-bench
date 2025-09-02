#!/usr/bin/env python3
"""Batch experiment example for comparing multiple models and cases."""

from psy_bench import Experiment
from psy_bench.cases import CASE_NAMES

def main():
    """Run a batch experiment comparing multiple models on selected cases."""
    
    # Configuration
    models_to_test = [
        "google/gemini-2.5-flash",
        "openai/gpt-4o-mini"
    ]
    
    # Select a subset of cases for demo (first 2 pairs)
    cases_to_test = [
        "Case 1.1: The Conduit [EXPLICIT]",
        "Case 1.2: The Conduit [IMPLICIT]", 
        "Case 2.1: The Digital Romance [EXPLICIT]",
        "Case 2.2: The Digital Romance [IMPLICIT]"
    ]
    
    print("🚀 Starting batch experiment")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Cases: {len(cases_to_test)} test cases")
    print(f"Total experiments: {len(models_to_test)} × {len(cases_to_test)} = {len(models_to_test) * len(cases_to_test)}")
    
    # Create experiment
    experiment = Experiment()
    
    try:
        # Run batch experiment
        results = experiment.run_batch(
            cases=cases_to_test,
            models=models_to_test,
            verbose=True
        )
        
        print("\\n" + "="*80)
        print("BATCH EXPERIMENT SUMMARY")
        print("="*80)
        
        # Create summary table
        print(f"{'Model':<30} {'Case':<40} {'DCS':<6} {'HES':<6} {'SIS':<6}")
        print("-" * 80)
        
        for result in results:
            model = result.metadata['target_model'].split('/')[-1]  # Just model name
            case = result.metadata['case_name'].split(':')[1].strip()  # Just case name
            dcs = f"{result.summary_stats['avg_dcs']:.2f}"
            hes = f"{result.summary_stats['avg_hes']:.2f}" 
            sis = str(result.summary_stats['total_sis'])
            
            print(f"{model:<30} {case:<40} {dcs:<6} {hes:<6} {sis:<6}")
        
        # Calculate model averages
        print("\\n" + "="*50)
        print("MODEL AVERAGES")
        print("="*50)
        
        model_stats = {}
        for result in results:
            model = result.metadata['target_model']
            if model not in model_stats:
                model_stats[model] = {'dcs_scores': [], 'hes_scores': [], 'sis_totals': []}
            
            model_stats[model]['dcs_scores'].append(result.summary_stats['avg_dcs'])
            model_stats[model]['hes_scores'].append(result.summary_stats['avg_hes'])
            model_stats[model]['sis_totals'].append(result.summary_stats['total_sis'])
        
        print(f"{'Model':<30} {'Avg DCS':<10} {'Avg HES':<10} {'Total SIS':<10}")
        print("-" * 60)
        
        for model, stats in model_stats.items():
            model_name = model.split('/')[-1]
            avg_dcs = sum(stats['dcs_scores']) / len(stats['dcs_scores'])
            avg_hes = sum(stats['hes_scores']) / len(stats['hes_scores']) 
            total_sis = sum(stats['sis_totals'])
            
            print(f"{model_name:<30} {avg_dcs:<10.2f} {avg_hes:<10.2f} {total_sis:<10}")
        
        print("\\n✅ Batch experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Batch experiment failed: {e}")


if __name__ == "__main__":
    main()