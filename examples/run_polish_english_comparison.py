"""Run the matched Polish-English Psychosis-Bench extension."""

from pathlib import Path

from psy_bench import ExportFormat, PsyBench


MODELS = [
    "meta-llama/llama-4-maverick",
    "google/gemini-2.5-pro",
    "openai/gpt-4o",
    "openai/o4-mini",
    "google/gemini-2.5-flash",
    "openai/gpt-5",
    "deepseek/deepseek-chat-v3.1",
    "anthropic/claude-sonnet-4",
]


def main() -> None:
    bench = PsyBench(case_languages=["en", "pl"])
    paired_cases = bench.get_paired_cases(languages=["en", "pl"])
    if not paired_cases:
        raise RuntimeError(
            "No matched English/Polish cases found. Populate data/test_cases_pl.json "
            "with cases whose source_case_id matches the English case id."
        )

    results = bench.run_language_comparison(models=MODELS, verbose=True)
    analyzer = bench.analyze(results)
    Path("outputs").mkdir(exist_ok=True)
    analyzer.export(ExportFormat.EXCEL, "outputs/polish_english_comparison.xlsx")

    deltas = analyzer.compare_languages("en", "pl")
    deltas.to_csv("outputs/polish_english_language_deltas.csv", index=False)


if __name__ == "__main__":
    main()
