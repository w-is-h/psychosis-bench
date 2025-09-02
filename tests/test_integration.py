#!/usr/bin/env python3
"""Real integration tests for psy-bench that make actual API calls."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from psy_bench import PsyBench, ExportFormat
from psy_bench.core.models import ScoreType


class TestRealIntegration:
    """Real integration tests that make actual API calls."""
    
    def test_basic_experiment_workflow(self):
        """Test basic experiment with real API calls."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Run a minimal experiment (just 1 turn to save API costs)
        result = bench.run(
            case="Case 1.1: The Conduit [EXPLICIT]",
            model="google/gemini-2.5-flash",
            start_turn=1,
            end_turn=1,
            verbose=False
        )
        
        # Verify basic structure
        assert result.model == "google/gemini-2.5-flash"
        assert result.test_case.name == "Case 1.1: The Conduit [EXPLICIT]"
        assert len(result.turns) == 1
        assert result.turns[0].number == 1
        assert result.turns[0].user_message
        assert result.turns[0].bot_response
        
        # Verify summary is calculated
        summary = result.summary
        assert "total_turns" in summary
        assert summary["total_turns"] == 1
        
        print(f"✅ Single experiment completed: {result.model} on {result.test_case.name}")
        print(f"   Bot response (first 100 chars): {result.turns[0].bot_response[:100]}...")
    
    def test_batch_processing(self):
        """Test batch processing with real API calls."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Run minimal batch (2 cases, 1 model to save costs)
        results = bench.run_batch(
            cases=[
                "Case 1.1: The Conduit [EXPLICIT]",
                "Case 1.2: The Conduit [IMPLICIT]"
            ],
            models=["google/gemini-2.5-flash"],
            verbose=False
        )
        
        # Should have 2 results (2 cases × 1 model)
        assert len(results) == 2
        
        # Check each result
        for result in results:
            assert result.model == "google/gemini-2.5-flash"
            assert len(result.turns) >= 1  # Should have at least 1 turn
            assert result.turns[0].bot_response
        
        # Check we got both cases
        case_names = [r.test_case.name for r in results]
        assert "Case 1.1: The Conduit [EXPLICIT]" in case_names
        assert "Case 1.2: The Conduit [IMPLICIT]" in case_names
        
        print(f"✅ Batch processing completed: {len(results)} experiments")
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing with real API calls."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Run minimal async batch
        results = await bench.run_batch_async(
            cases=["Case 1.1: The Conduit [EXPLICIT]"],
            models=["google/gemini-2.5-flash", "openai/gpt-4o-mini"],
            max_concurrent=2,
            verbose=False
        )
        
        # Should have 2 results (1 case × 2 models)
        assert len(results) == 2
        
        # Check we got both models
        models = [r.model for r in results]
        assert "google/gemini-2.5-flash" in models
        assert "openai/gpt-4o-mini" in models
        
        # Each result should have valid data
        for result in results:
            assert result.test_case.name == "Case 1.1: The Conduit [EXPLICIT]"
            assert len(result.turns) >= 1  # Should have at least 1 turn
            assert result.turns[0].bot_response
        
        print(f"✅ Async batch processing completed: {len(results)} experiments")
    
    def test_scoring_system(self):
        """Test that scoring system works with real API calls."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Run experiment with scoring
        result = bench.run(
            case="Case 1.1: The Conduit [EXPLICIT]",
            model="google/gemini-2.5-flash",
            start_turn=1,
            end_turn=2,  # Need at least 2 turns for meaningful scoring
            verbose=False
        )
        
        # Check that turns have some scores
        scored_turns = [turn for turn in result.turns if turn.scores]
        assert len(scored_turns) > 0, "At least some turns should have scores"
        
        # Check score types are present
        all_scores = {}
        for turn in result.turns:
            for score_type, score in turn.scores.items():
                if score is not None:
                    all_scores[score_type] = score
        
        print(f"✅ Scoring completed. Found scores: {list(all_scores.keys())}")
        
        # Verify summary calculations
        summary = result.summary
        if ScoreType.DCS in all_scores:
            assert "avg_dcs" in summary
        if ScoreType.HES in all_scores:
            assert "avg_hes" in summary
        if ScoreType.SIS in all_scores:
            assert "total_sis" in summary
    
    def test_analysis_and_export(self):
        """Test analysis and export functionality."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Run a couple experiments
        results = bench.run_batch(
            cases=["Case 1.1: The Conduit [EXPLICIT]"],
            models=["google/gemini-2.5-flash", "openai/gpt-4o-mini"],
            verbose=False
        )
        
        # Analyze results
        analyzer = bench.analyze(results, print_summary=False)
        summary = analyzer.get_summary()
        
        # Check analysis structure
        assert "overview" in summary
        assert "by_model" in summary
        assert "by_case" in summary
        assert "by_theme" in summary
        
        # Check overview stats
        overview = summary["overview"]
        assert overview["total_experiments"] == 2
        assert overview["unique_models"] == 2
        assert overview["unique_cases"] == 1
        
        # Test export to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            analyzer.export(ExportFormat.JSON, json_path)
            
            # Verify exported file
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "metadata" in exported_data
            assert "summary" in exported_data
            assert "results" in exported_data
            assert len(exported_data["results"]) == 2
            
        finally:
            Path(json_path).unlink(missing_ok=True)
        
        print(f"✅ Analysis and export completed for {len(results)} results")
    
    def test_custom_case_loading(self):
        """Test loading and running custom test cases."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Create a custom test case file
        custom_case_data = {
            "cases": [
                {
                    "id": "test_simple",
                    "name": "Simple Test Case",
                    "theme": "Test Theme",
                    "condition": "Explicit",
                    "harm_type": "None",
                    "prompts": [
                        "Hello, how are you today?",
                        "Can you tell me about the weather?"
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_case_data, f, indent=2)
            custom_file_path = f.name
        
        try:
            # Load custom cases
            custom_cases = bench.load_cases_from_file(custom_file_path)
            assert len(custom_cases) == 1
            assert custom_cases[0].name == "Simple Test Case"
            
            # Run experiment with custom case
            result = bench.run(
                case=custom_cases[0],
                model="google/gemini-2.5-flash",
                start_turn=1,
                end_turn=1,
                verbose=False
            )
            
            assert result.test_case.name == "Simple Test Case"
            assert len(result.turns) == 1
            
            print(f"✅ Custom case loading and execution completed")
            
        finally:
            Path(custom_file_path).unlink(missing_ok=True)
    
    def test_theme_filtering(self):
        """Test filtering cases by theme."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Get all available themes
        themes = bench.get_themes()
        assert len(themes) > 0
        
        # Pick first theme and get cases for it
        test_theme = themes[0]
        theme_cases = [
            case for case in bench.list_cases()
            if bench.get_case_theme(case) == test_theme
        ]
        
        assert len(theme_cases) > 0, f"Should have cases for theme: {test_theme}"
        
        # Run one case from this theme
        result = bench.run(
            case=theme_cases[0],
            model="google/gemini-2.5-flash",
            start_turn=1,
            end_turn=1,
            verbose=False
        )
        
        assert result.test_case.theme == test_theme
        
        print(f"✅ Theme filtering completed: {test_theme} has {len(theme_cases)} cases")
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        try:
            bench = PsyBench()
        except ValueError as e:
            pytest.skip(f"No API key available: {e}")
        
        # Test invalid case name
        with pytest.raises(Exception):  # Should raise some kind of error
            bench.run(
                case="Non-existent Case",
                model="google/gemini-2.5-flash",
                start_turn=1,
                end_turn=1
            )
        
        # Test invalid model (this should fail gracefully, not crash)
        try:
            result = bench.run(
                case="Case 1.1: The Conduit [EXPLICIT]",
                model="non-existent/model",
                start_turn=1,
                end_turn=1,
                verbose=False
            )
            # If it doesn't raise an error, check that error is handled gracefully
            assert "ERROR:" in result.turns[0].bot_response
        except Exception:
            # This is also acceptable - API error for invalid model
            pass
        
        print("✅ Error handling tested")


class TestConfigurationAndSetup:
    """Test configuration and setup without API calls."""
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails properly without API key."""
        # This test should work regardless of API key availability
        # We'll temporarily override both environment and settings
        import os
        from unittest.mock import patch
        
        original_key = os.environ.get('OPENROUTER_API_KEY')
        
        try:
            # Remove API key from environment
            if 'OPENROUTER_API_KEY' in os.environ:
                del os.environ['OPENROUTER_API_KEY']
            
            # Mock the config to return None for API key
            with patch('psy_bench.api.OPENROUTER_API_KEY', None):
                with pytest.raises(ValueError, match="OpenRouter API key is required"):
                    PsyBench()
                
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ['OPENROUTER_API_KEY'] = original_key
    
    def test_case_listing_without_api(self):
        """Test that case listing works without API calls."""
        try:
            bench = PsyBench()
        except ValueError:
            # If no API key, we can still test case loading directly
            from psy_bench.core.cases import CaseLoader
            cases = CaseLoader.load_bundled_cases()
            case_names = [case.name for case in cases]
            
            assert len(case_names) >= 1
            assert "Case 1.1: The Conduit [EXPLICIT]" in case_names
            return
        
        # If we have API key, test through PsyBench interface
        cases = bench.list_cases()
        themes = bench.get_themes()
        
        assert len(cases) >= 1
        assert len(themes) >= 1