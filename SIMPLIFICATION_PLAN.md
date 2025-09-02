# Psy-Bench Simplification Plan

## Executive Summary

This plan outlines specific steps to simplify the psy-bench library, reducing code complexity by ~40% while maintaining all functionality and improving usability.

## 1. API Client Consolidation

### Current Issues
- Separate sync (`api_client.py`) and async (`async_client.py`) clients with 70% duplicate code
- Redundant retry logic, error handling, and request formatting

### Proposed Solution
```python
# psy_bench/core/client.py
class OpenRouterClient:
    """Unified client supporting both sync and async operations."""
    
    def __init__(self, api_key: str, **config):
        self.api_key = api_key
        self.config = config
    
    def chat(self, model: str, messages: List[ChatMessage]) -> ChatResponse:
        """Synchronous chat request."""
        return self._request(model, messages)
    
    async def achat(self, model: str, messages: List[ChatMessage]) -> ChatResponse:
        """Asynchronous chat request."""
        return await self._arequest(model, messages)
    
    # Shared logic for both sync/async
    def _prepare_request(self, model, messages):
        """Common request preparation logic."""
        pass
```

### Benefits
- 50% less code to maintain
- Consistent behavior between sync/async
- Single point for API changes

## 2. Experiment Runner Simplification

### Current Issues
- `Experiment` class has 568 lines handling:
  - Single experiments
  - Batch processing
  - Async operations
  - Scoring orchestration
  - Statistics calculation

### Proposed Solution
```python
# psy_bench/core/runner.py
class ExperimentRunner:
    """Focused class for running single experiments."""
    
    def __init__(self, client: OpenRouterClient, scorer: Scorer):
        self.client = client
        self.scorer = scorer
    
    def run(self, test_case: TestCase, config: RunConfig) -> ExperimentResult:
        """Run a single experiment."""
        pass

# psy_bench/core/batch.py
class BatchRunner:
    """Handles batch experiment execution."""
    
    def __init__(self, runner: ExperimentRunner):
        self.runner = runner
    
    def run_batch(self, cases: List[TestCase], models: List[str]) -> List[ExperimentResult]:
        """Run multiple experiments."""
        pass
    
    async def run_batch_async(self, cases, models) -> List[ExperimentResult]:
        """Run experiments concurrently."""
        pass
```

### Benefits
- Each class has single responsibility
- Easier to test and maintain
- More flexible composition

## 3. Unified Scoring System

### Current Issues
- Three separate scorer classes with similar structure
- Redundant prompt generation and parsing logic
- Mixed LLM-based and keyword-based scoring

### Proposed Solution
```python
# psy_bench/core/scoring.py
class Scorer:
    """Unified scorer for all metrics."""
    
    def __init__(self, client: OpenRouterClient, judge_model: str):
        self.client = client
        self.judge_model = judge_model
        self.configs = {
            ScoreType.DCS: DCSConfig(),
            ScoreType.HES: HESConfig(),
            ScoreType.SIS: SISConfig()
        }
    
    def score(self, conversation: List[Turn], score_type: ScoreType, **kwargs) -> ScoreResult:
        """Score conversation for specified metric."""
        config = self.configs[score_type]
        
        if config.use_llm:
            return self._llm_score(conversation, config, **kwargs)
        else:
            return self._keyword_score(conversation, config)
```

### Benefits
- 60% less scoring code
- Consistent scoring interface
- Easier to add new metrics

## 4. Test Case Externalization

### Current Issues
- 469 lines of hardcoded test cases in `archetypal.py`
- Difficult to modify or add cases
- Mixed data and logic

### Proposed Solution
```python
# data/test_cases.json
{
  "version": "1.0",
  "cases": [
    {
      "id": "conduit_explicit",
      "name": "Case 1.1: The Conduit [EXPLICIT]",
      "theme": "Grandiose Delusions",
      "condition": "Explicit",
      "harm_type": "Self-Harm (Jumping from height)",
      "phases": {
        "initial_engagement": [...],
        "pattern_seeking": [...],
        "delusion_solidification": [...],
        "behavioral_enactment": [...]
      }
    }
  ]
}

# psy_bench/cases/loader.py
class CaseLoader:
    """Load test cases from JSON files."""
    
    @staticmethod
    def load_bundled_cases() -> List[TestCase]:
        """Load the standard test cases."""
        return CaseLoader.load_from_file("data/test_cases.json")
    
    @staticmethod
    def load_from_file(path: str) -> List[TestCase]:
        """Load cases from any JSON file."""
        pass
```

### Benefits
- Separate data from code
- Easy to version and modify cases
- Support for custom case files

## 5. Simplified Analysis Module

### Current Issues
- Separate `reporter.py` and `exporters.py` with overlapping functionality
- Redundant statistics calculation
- Complex class hierarchy for exporters

### Proposed Solution
```python
# psy_bench/analysis.py
class ResultAnalyzer:
    """Unified analysis and export functionality."""
    
    def __init__(self, results: List[ExperimentResult]):
        self.results = results
        self._stats = None
    
    def get_summary(self) -> Summary:
        """Get analysis summary."""
        if not self._stats:
            self._stats = self._calculate_stats()
        return self._stats
    
    def export(self, format: ExportFormat, path: str, **options):
        """Export results in specified format."""
        exporters = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.EXCEL: self._export_excel
        }
        return exporters[format](path, **options)
```

### Benefits
- Single source of truth for statistics
- Simpler export interface
- 30% less code

## 6. Streamlined Examples

### Current Issues
- 5 example files with significant overlap
- Unclear which to use when
- Maintenance burden

### Proposed Solution
Keep only 2 examples:

```python
# examples/quickstart.py
"""Basic usage example - run a single experiment."""
from psy_bench import PsyBench

# Simple, clear example
bench = PsyBench()
result = bench.run("Case 1.1: The Conduit [EXPLICIT]")
print(result.summary)

# examples/advanced.py
"""Advanced usage - batch processing and custom configuration."""
# Demonstrates batch, async, custom cases, and export
```

### Benefits
- Clear progression from simple to advanced
- Less confusion for new users
- Easier to maintain

## 7. Simplified Public API

### Current Issues
- Multiple entry points (Experiment, various runners)
- Unclear what to import and use
- Complex configuration

### Proposed Solution
```python
# psy_bench/__init__.py
class PsyBench:
    """Main entry point for the library."""
    
    def __init__(self, api_key: str = None, **config):
        """Initialize with optional configuration."""
        self.config = Config(api_key=api_key, **config)
        self._client = None
        self._runner = None
    
    def run(self, case: str, model: str = None) -> Result:
        """Run a single experiment."""
        return self._get_runner().run(case, model or self.config.default_model)
    
    def run_batch(self, cases: List[str], models: List[str]) -> BatchResult:
        """Run multiple experiments."""
        return BatchRunner(self._get_runner()).run_batch(cases, models)
    
    def analyze(self, results: List[Result]) -> Analysis:
        """Analyze experiment results."""
        return ResultAnalyzer(results).get_summary()

# Usage becomes:
from psy_bench import PsyBench

bench = PsyBench(api_key="...")
result = bench.run("Case 1.1: The Conduit [EXPLICIT]")
```

### Benefits
- Single, clear entry point
- Intuitive method names
- Progressive disclosure of complexity

## Implementation Priority

1. **Phase 1** (High Impact, Low Risk):
   - Externalize test cases to JSON
   - Consolidate examples
   - Create unified API entry point

2. **Phase 2** (Medium Impact, Medium Risk):
   - Merge API clients
   - Simplify scoring system
   - Consolidate analysis modules

3. **Phase 3** (Lower Impact, Higher Risk):
   - Refactor Experiment class
   - Optimize async processing

## Expected Outcomes

- **40% reduction** in total lines of code
- **60% reduction** in API surface area
- **Improved maintainability** through single-responsibility principles
- **Better developer experience** with clearer entry points
- **Preserved functionality** - all current features remain available

## Migration Guide

For existing users, provide a migration guide showing:
- Old import paths → New import paths
- Deprecated methods → New methods
- Configuration changes
