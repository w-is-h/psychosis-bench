#!/usr/bin/env python3
"""Async batch processing demo with real API calls."""

import time
import asyncio
from psy_bench import Experiment
from psy_bench.cases import CASE_NAMES
from psy_bench.config.settings import settings

async def async_demo():
    """Run actual async batch experiment."""
    
    # Small demo: 2 cases, 1 model = 2 experiments  
    test_cases = CASE_NAMES[:10]
    models = ["google/gemini-2.5-flash"]
    
    print(f"🚀 Running {len(models)} model × {len(test_cases)} cases = {len(models) * len(test_cases)} experiments")
    
    experiment = Experiment()
    start_time = time.time()
    
    try:
        results = await experiment.run_batch_async(
            cases=test_cases,
            models=models,
            max_concurrent=settings.max_concurrent_requests,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\\n✅ Completed {len(results)} experiments in {elapsed:.1f}s")
        
        for result in results:
            model_name = result.metadata['target_model'].split('/')[-1]
            case_name = result.metadata['case_name'].split(':')[1].strip()
            turns = result.summary_stats['total_turns']
            print(f"   {model_name}: {case_name} ({turns} turns)")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return []

def sync_comparison():
    """Run one sequential experiment for timing comparison."""
    experiment = Experiment()
    test_case = CASE_NAMES[0]
    
    start_time = time.time()
    
    try:
        result = experiment.run_case(case=test_case, verbose=False)
        elapsed = time.time() - start_time
        print(f"Sequential baseline: {elapsed:.1f}s")
        return elapsed
        
    except Exception as e:
        print(f"Sequential failed: {e}")
        return None

def main():
    """Run the demo."""
    print("⚡ Psy-Bench Async Demo")
    print("=" * 30)
    
    sequential_time = sync_comparison()
    print()
    
    async_results = asyncio.run(async_demo())
    
    if sequential_time and async_results:
        print(f"\\n📊 Async enables concurrent processing for larger batches")

if __name__ == "__main__":
    main()
