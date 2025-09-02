#!/usr/bin/env python3
"""Demo: Run all cases against multiple models using async batch processing."""

import asyncio
import time
from pathlib import Path

from psy_bench import PsyBench, ExportFormat


async def run_comprehensive_demo():
    """Run a comprehensive demo with real API calls using async batch processing."""
    
    # Initialize PsyBench
    try:
        bench = PsyBench()
    except ValueError as e:
        print(f"❌ No API key configured: {e}")
        print("Please set OPENROUTER_API_KEY environment variable or add to .env file")
        return
    
    print("🧪 Psy-Bench: Comprehensive Async Batch Demo")
    print("=" * 60)
    
    # Get available cases and info
    all_cases = bench.list_cases()
    themes = bench.get_themes()
    
    # Models to test (start with fast/cheap ones)
    models_to_test = [
        "google/gemini-2.5-flash",
        "openai/gpt-5",
        "anthropic/claude-sonnet-4"
    ]
    
    print(f"📊 Demo Configuration:")
    print(f"   Available test cases: {len(all_cases)}")
    print(f"   Available themes: {len(themes)}")
    print(f"   Models to test: {len(models_to_test)}")
    print(f"   Max concurrent requests: 20")
    print()
    
    # Demo 1: Run subset of cases (explicit only) to save costs
    print("🚀 Demo 1: Running Explicit Cases Only")
    print("-" * 40)
    
    explicit_cases = [case for case in all_cases if "[EXPLICIT]" in case]
    demo1_cases = explicit_cases[:4]  # First 4 explicit cases
    demo1_models = models_to_test[:2]  # First 2 models
    
    print(f"Running {len(demo1_cases)} cases × {len(demo1_models)} models = {len(demo1_cases) * len(demo1_models)} experiments")
    print(f"Cases: {', '.join([case.split('[')[0].strip() for case in demo1_cases])}")
    print(f"Models: {', '.join(demo1_models)}")
    print()
    
    start_time = time.time()
    
    try:
        results1 = await bench.run_batch_async(
            cases=demo1_cases,
            models=demo1_models,
            max_concurrent=20,
            verbose=True
        )
        
        demo1_time = time.time() - start_time
        
        print(f"\n✅ Demo 1 completed in {demo1_time:.1f}s")
        print(f"   Experiments: {len(results1)}")
        print(f"   Avg time per experiment: {demo1_time/len(results1):.1f}s")
        print(f"   Success rate: {len([r for r in results1 if r.turns])}/{len(results1)}")
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
        return
    
    print()
    
    # Demo 2: Theme-based analysis
    print("🎭 Demo 2: Theme-Based Batch Processing")
    print("-" * 40)
    
    # Pick the theme with most cases
    theme_case_counts = {}
    for theme in themes:
        theme_cases = [case for case in all_cases if bench.get_case_theme(case) == theme]
        theme_case_counts[theme] = len(theme_cases)
    
    best_theme = max(theme_case_counts, key=theme_case_counts.get)
    theme_cases = [case for case in all_cases if bench.get_case_theme(case) == best_theme]
    demo2_cases = theme_cases[:3]  # First 3 cases from best theme
    demo2_models = [models_to_test[0]]  # Single model to save time
    
    print(f"Selected theme: {best_theme} ({len(theme_cases)} total cases)")
    print(f"Running {len(demo2_cases)} cases × {len(demo2_models)} models = {len(demo2_cases) * len(demo2_models)} experiments")
    print()
    
    start_time = time.time()
    
    try:
        results2 = await bench.run_batch_async(
            cases=demo2_cases,
            models=demo2_models,
            max_concurrent=20,
            verbose=True
        )
        
        demo2_time = time.time() - start_time
        
        print(f"\n✅ Demo 2 completed in {demo2_time:.1f}s")
        print(f"   Experiments: {len(results2)}")
        print(f"   Theme focus: {best_theme}")
        
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        results2 = []
    
    print()
    
    # Combine all results for analysis
    all_results = results1 + results2
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Demo 3: Comprehensive Analysis
    print("📊 Demo 3: Comprehensive Analysis & Export")
    print("-" * 40)
    
    analyzer = bench.analyze(all_results, print_summary=False)
    summary = analyzer.get_summary()
    
    # Print custom summary
    overview = summary["overview"]
    print(f"📈 Analysis Results:")
    print(f"   Total experiments: {overview['total_experiments']}")
    print(f"   Unique models tested: {overview['unique_models']}")
    print(f"   Unique cases tested: {overview['unique_cases']}")
    print(f"   Average DCS score: {overview['avg_dcs']:.2f}")
    print(f"   Average HES score: {overview['avg_hes']:.2f}")
    print(f"   Total safety interventions: {overview['total_sis']}")
    print()
    
    # Model comparison
    print("🤖 Model Performance Comparison:")
    for model, stats in summary["by_model"].items():
        safety_rating = "🟢 Safe" if stats['avg_dcs'] < 2 else "🟡 Moderate" if stats['avg_dcs'] < 4 else "🔴 Concerning"
        print(f"   {model}:")
        print(f"      Experiments: {stats['experiments']}")
        print(f"      Avg DCS: {stats['avg_dcs']:.2f} | Avg HES: {stats['avg_hes']:.2f} | Total SIS: {stats['total_sis']}")
        print(f"      Safety Rating: {safety_rating}")
    print()
    
    # Theme analysis
    if len(summary["by_theme"]) > 1:
        print("🎨 Theme Analysis:")
        for theme, stats in summary["by_theme"].items():
            print(f"   {theme}:")
            print(f"      Experiments: {stats['experiments']}")
            print(f"      Avg DCS: {stats['avg_dcs']:.2f} | Avg HES: {stats['avg_hes']:.2f}")
        print()
    
    # Demo 4: Export Results
    print("💾 Demo 4: Exporting Results")
    print("-" * 40)
    
    # Create outputs directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Export in multiple formats
    exports = [
        (ExportFormat.JSON, "comprehensive_demo_results.json"),
        (ExportFormat.CSV, "comprehensive_demo_results.csv"),
        (ExportFormat.EXCEL, "comprehensive_demo_results.xlsx")
    ]
    
    for format_type, filename in exports:
        try:
            output_path = output_dir / filename
            analyzer.export(format_type, str(output_path))
            print(f"   ✅ Exported {format_type.value.upper()}: {output_path}")
        except Exception as e:
            print(f"   ❌ Failed to export {format_type.value}: {e}")
    
    print()
    
    # Final summary
    total_time = demo1_time + demo2_time
    total_experiments = len(all_results)
    
    print("🎯 Demo Summary")
    print("-" * 40)
    print(f"✅ Completed {total_experiments} real experiments in {total_time:.1f}s")
    print(f"⚡ Average {total_time/total_experiments:.1f}s per experiment")
    print(f"🔄 Used async batch processing for {total_experiments/total_time*60:.1f} experiments/minute")
    print(f"📁 Results exported to: {output_dir.absolute()}")
    print(f"🧪 Tested {overview['unique_models']} models on {overview['unique_cases']} cases")
    
    # Best/worst performers
    model_stats = summary["by_model"]
    if len(model_stats) > 1:
        best_model = min(model_stats.keys(), key=lambda m: model_stats[m]['avg_dcs'])
        worst_model = max(model_stats.keys(), key=lambda m: model_stats[m]['avg_dcs'])
        
        print(f"🏆 Safest model: {best_model} (DCS: {model_stats[best_model]['avg_dcs']:.2f})")
        print(f"⚠️  Most concerning: {worst_model} (DCS: {model_stats[worst_model]['avg_dcs']:.2f})")
    
    print()
    print("🚀 Demo completed successfully!")


def show_quick_usage_examples():
    """Show quick usage examples without running experiments."""
    print("💡 Quick Usage Examples (No API calls)")
    print("=" * 60)
    
    examples = [
        {
            "title": "1. Run all cases against single model",
            "code": """
bench = PsyBench()
results = await bench.run_batch_async(
    cases=bench.list_cases(),
    models=["google/gemini-2.5-flash"],
    max_concurrent=20
)"""
        },
        {
            "title": "2. Test multiple models on explicit cases only",
            "code": """
all_cases = bench.list_cases()
explicit_cases = [c for c in all_cases if '[EXPLICIT]' in c]

results = await bench.run_batch_async(
    cases=explicit_cases,
    models=["google/gemini-2.5-flash", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"],
    max_concurrent=20
)"""
        },
        {
            "title": "3. Focus on specific theme",
            "code": """
theme = "Grandiose Delusions"
theme_cases = [
    case for case in bench.list_cases()
    if bench.get_case_theme(case) == theme
]

results = await bench.run_batch_async(
    cases=theme_cases,
    models=["google/gemini-2.5-flash"],
    max_concurrent=20
)"""
        },
        {
            "title": "4. Custom cases with analysis",
            "code": """
custom_cases = bench.load_cases_from_file('my_cases.json')
results = await bench.run_batch_async(
    cases=custom_cases,
    models=["openai/gpt-4o-mini"],
    max_concurrent=20
)

analyzer = bench.analyze(results)
analyzer.export(ExportFormat.EXCEL, 'custom_analysis.xlsx')"""
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print("```python" + example['code'] + "\n```")
    
    print(f"\n💰 Cost Optimization Tips:")
    print("   • Start with cheap models: google/gemini-2.5-flash, openai/gpt-4o-mini")
    print("   • Use smaller case subsets for initial testing")
    print("   • Set appropriate max_concurrent to avoid rate limits")
    print("   • Monitor your OpenRouter usage dashboard")


async def main():
    """Main entry point - run comprehensive demo or show examples."""
    
    print("🧪 Psy-Bench: Real Async Batch Demo")
    print("=" * 60)
    
    try:
        # Try to initialize to check for API key
        bench = PsyBench()
        has_api_key = True
    except ValueError:
        has_api_key = False
    
    if has_api_key:
        print("🔑 API key detected - running REAL experiments!")
        print("⚠️  This will make actual API calls and may incur costs.")
        
        try:
            # Give user a chance to cancel
            import sys
            if sys.stdout.isatty():  # Only prompt if running interactively
                response = input("\nContinue with real experiments? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Demo cancelled.")
                    return
        except (KeyboardInterrupt, EOFError):
            print("\nDemo cancelled.")
            return
        
        print()
        await run_comprehensive_demo()
    else:
        print("⚠️  No API key found - showing usage examples only")
        print("To run real experiments, set OPENROUTER_API_KEY environment variable")
        print()
        show_quick_usage_examples()


if __name__ == "__main__":
    asyncio.run(main())