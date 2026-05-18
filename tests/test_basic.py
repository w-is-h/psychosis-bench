#!/usr/bin/env python3
"""Basic tests for psy-bench."""

import pytest
from unittest.mock import patch

from psy_bench import PsyBench, TestCase, ExperimentResult
from psy_bench.core.client import ChatMessage, ChatResponse
from psy_bench.core.models import ScoreType, Turn


class TestPsyBench:
    """Test the main PsyBench API."""
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch('psy_bench.api.OPENROUTER_API_KEY', None):
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                PsyBench()
    
    def test_initialization_with_api_key(self):
        """Test successful initialization with API key."""
        bench = PsyBench(api_key="test-key")
        assert bench.api_key == "test-key"
        assert bench._client is not None
        assert bench._scorer is not None
        assert bench._runner is not None
    
    def test_list_cases(self):
        """Test listing available cases."""
        bench = PsyBench(api_key="test-key")
        cases = bench.list_cases()
        
        assert len(cases) >= 1  # Should have at least 1 case
        assert all(isinstance(case, str) for case in cases)
        assert "Case 1.1: The Conduit [EXPLICIT]" in cases

    def test_default_language_metadata(self):
        """Test that bundled English cases expose language metadata."""
        bench = PsyBench(api_key="test-key")

        assert bench.get_languages() == ["en"]
        assert len(bench.list_cases_by_language("en")) == len(bench.list_cases())
    
    def test_get_themes(self):
        """Test getting available themes."""
        bench = PsyBench(api_key="test-key")
        themes = bench.get_themes()
        
        assert len(themes) > 0
        assert "Grandiose Delusions" in themes
        assert all(isinstance(theme, str) for theme in themes)
    
    def test_get_case_theme(self):
        """Test getting theme for a specific case."""
        bench = PsyBench(api_key="test-key")
        
        theme = bench.get_case_theme("Case 1.1: The Conduit [EXPLICIT]")
        assert theme == "Grandiose Delusions"
        
        theme = bench.get_case_theme("Non-existent case")
        assert theme is None
    
    @patch('psy_bench.core.client.OpenRouterClient.chat')
    @patch('psy_bench.core.client.OpenRouterClient.judge')
    def test_run_single_experiment(self, mock_judge, mock_chat):
        """Test running a single experiment."""
        # Mock API responses
        mock_chat.return_value = ChatResponse(
            content="I understand your concerns.",
            model="test-model"
        )
        mock_judge.return_value = "1"  # Neutral score
        
        bench = PsyBench(api_key="test-key")
        result = bench.run(
            "Case 1.1: The Conduit [EXPLICIT]",
            model="test-model",
            start_turn=1,
            end_turn=3,
            verbose=False
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.model == "test-model"
        assert len(result.turns) == 3
        assert result.test_case.name == "Case 1.1: The Conduit [EXPLICIT]"
    
    @patch('psy_bench.core.client.OpenRouterClient.chat')
    @patch('psy_bench.core.client.OpenRouterClient.judge')
    def test_run_batch(self, mock_judge, mock_chat):
        """Test running batch experiments."""
        mock_chat.return_value = ChatResponse(
            content="Test response",
            model="test-model"
        )
        mock_judge.return_value = "1"
        
        bench = PsyBench(api_key="test-key")
        results = bench.run_batch(
            cases=["Case 1.1: The Conduit [EXPLICIT]"],
            models=["model1", "model2"],
            verbose=False
        )
        
        assert len(results) == 2  # 1 case × 2 models
        assert all(isinstance(r, ExperimentResult) for r in results)
    
    def test_analyze_results(self):
        """Test analyzing experiment results."""
        bench = PsyBench(api_key="test-key")
        
        # Create mock results
        mock_result = ExperimentResult(
            test_case=TestCase(
                id="test",
                name="Test Case",
                theme="Test Theme",
                condition="Explicit",
                harm_type="Test Harm",
                prompts=["test"] * 12
            ),
            model="test-model",
            turns=[
                Turn(
                    number=1,
                    user_message="test",
                    bot_response="response",
                    scores={ScoreType.DCS: 1}
                )
            ]
        )
        
        analyzer = bench.analyze([mock_result], print_summary=False)
        
        assert analyzer is not None
        summary = analyzer.get_summary()
        assert summary['overview']['total_experiments'] == 1
        assert summary['by_language']['en']['experiments'] == 1

    def test_language_comparison_analysis(self):
        """Test matched language delta analysis."""
        bench = PsyBench(api_key="test-key")

        english_case = TestCase(
            id="case_1",
            name="Case 1",
            theme="Test Theme",
            condition="Explicit",
            harm_type="Test Harm",
            prompts=["test"] * 12,
            language="en",
        )
        polish_case = TestCase(
            id="case_1_pl",
            source_case_id="case_1",
            name="Case 1 PL",
            theme="Test Theme",
            condition="Explicit",
            harm_type="Test Harm",
            prompts=["test"] * 12,
            language="pl",
            translation_status="validated",
        )
        results = [
            ExperimentResult(
                test_case=english_case,
                model="test-model",
                turns=[
                    Turn(number=7, user_message="x", bot_response="y", scores={ScoreType.DCS: 0, ScoreType.HES: 0, ScoreType.SIS: 1})
                ],
            ),
            ExperimentResult(
                test_case=polish_case,
                model="test-model",
                turns=[
                    Turn(number=7, user_message="x", bot_response="y", scores={ScoreType.DCS: 2, ScoreType.HES: 1, ScoreType.SIS: 0})
                ],
            ),
        ]

        analyzer = bench.analyze(results, print_summary=False)
        deltas = analyzer.compare_languages("en", "pl")

        assert len(deltas) == 1
        row = deltas.iloc[0]
        assert row["source_case_id"] == "case_1"
        assert row["delta_avg_dcs"] == 2
        assert row["delta_avg_hes"] == 1
        assert row["delta_total_sis"] == -1


class TestClient:
    """Test the unified client."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        from psy_bench.core.client import OpenRouterClient
        
        client = OpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.timeout == 30
        assert client.max_retries == 3
    
    def test_prepare_request(self):
        """Test request preparation."""
        from psy_bench.core.client import OpenRouterClient
        
        client = OpenRouterClient(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        headers, payload = client._prepare_request("test-model", messages)
        
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 1


class TestScoring:
    """Test the scoring system."""
    
    def test_scoring_configs(self):
        """Test that scoring configurations are properly defined."""
        from psy_bench.core.scoring import SCORING_CONFIGS, ScoreType
        
        assert ScoreType.DCS in SCORING_CONFIGS
        assert ScoreType.HES in SCORING_CONFIGS
        assert ScoreType.SIS in SCORING_CONFIGS
        
        # Check that configs exist and have basic info
        for score_type in [ScoreType.DCS, ScoreType.HES, ScoreType.SIS]:
            config = SCORING_CONFIGS[score_type]
            assert isinstance(config, dict)
            assert "name" in config or "description" in config
    
    @patch('psy_bench.core.client.OpenRouterClient.chat')
    def test_score_turn(self, mock_chat):
        """Test scoring a single turn."""
        from psy_bench.core.scoring import Scorer
        from psy_bench.core.client import OpenRouterClient, ChatResponse
        
        mock_chat.return_value = ChatResponse(content="1", model="test-model")
        
        client = OpenRouterClient(api_key="test-key")
        scorer = Scorer(client)
        
        conversation = [
            Turn(number=4, user_message="test", bot_response="response")
        ]
        
        result = scorer.score_turn(conversation, ScoreType.DCS)
        
        assert result.score >= 0  # Should be a valid score
        assert result.score_type == ScoreType.DCS


class TestCaseLoader:
    """Test case loading functionality."""
    
    def test_load_bundled_cases(self):
        """Test loading bundled test cases."""
        from psy_bench.core.cases import CaseLoader
        
        cases = CaseLoader.load_bundled_cases()
        
        assert len(cases) >= 1
        assert all(hasattr(case, 'name') for case in cases)
        assert all(hasattr(case, 'prompts') for case in cases)
        assert all(len(case.prompts) == 12 for case in cases)
    
    def test_get_case_by_name(self):
        """Test finding case by name."""
        from psy_bench.core.cases import CaseLoader
        
        case = CaseLoader.get_case_by_name("Case 1.1: The Conduit [EXPLICIT]")
        assert case is not None
        assert case.theme == "Grandiose Delusions"
        
        case = CaseLoader.get_case_by_name("Non-existent")
        assert case is None
    
    def test_filter_by_condition(self):
        """Test filtering cases by condition."""
        from psy_bench.core.cases import CaseLoader
        
        explicit = CaseLoader.get_explicit_cases()
        implicit = CaseLoader.get_implicit_cases()
        
        assert len(explicit) >= 1
        assert len(implicit) >= 1
        assert all(case.condition == "Explicit" for case in explicit)
        assert all(case.condition == "Implicit" for case in implicit)

    def test_get_paired_cases(self):
        """Test grouping cases by source id and language."""
        from psy_bench.core.cases import CaseLoader

        english_case = TestCase(
            id="case_1",
            name="Case 1",
            theme="Test Theme",
            condition="Explicit",
            harm_type="Test Harm",
            prompts=["test"] * 12,
            language="en",
        )
        polish_case = TestCase(
            id="case_1_pl",
            source_case_id="case_1",
            name="Case 1 PL",
            theme="Test Theme",
            condition="Explicit",
            harm_type="Test Harm",
            prompts=["test"] * 12,
            language="pl",
        )

        paired = CaseLoader.get_paired_cases([english_case, polish_case], languages=["en", "pl"])

        assert list(paired) == ["case_1"]
        assert paired["case_1"]["en"] == english_case
        assert paired["case_1"]["pl"] == polish_case


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
