"""Basic tests for psy-bench library."""

import pytest
from unittest.mock import Mock, patch

from psy_bench.cases.archetypal import get_case_by_name, ARCHETYPAL_CASES
from psy_bench.core.api_client import OpenRouterClient, ChatMessage
from psy_bench.core.scoring import DelusionConfirmationScorer, ConversationTurn
from psy_bench.config.settings import Settings


def test_case_loading():
    """Test that archetypal cases can be loaded correctly."""
    # Test we have the expected number of cases
    assert len(ARCHETYPAL_CASES) == 16  # 8 pairs × 2 (explicit/implicit)
    
    # Test getting a case by name
    case = get_case_by_name("Case 1.1: The Conduit [EXPLICIT]")
    assert case is not None
    assert case.name == "Case 1.1: The Conduit [EXPLICIT]"
    assert case.theme == "Grandiose Delusions"
    assert case.condition == "Explicit"
    assert len(case.prompts) == 12  # 4 phases × 3 prompts each
    
    # Test non-existent case
    case = get_case_by_name("Non-existent Case")
    assert case is None


def test_settings():
    """Test settings configuration."""
    settings = Settings()
    
    # Test that default values are set (not testing specific values)
    assert settings.default_target_model is not None
    assert len(settings.default_target_model) > 0
    assert "/" in settings.default_target_model  # Should be in format "provider/model"
    
    assert settings.default_judge_model is not None
    assert len(settings.default_judge_model) > 0
    assert "/" in settings.default_judge_model  # Should be in format "provider/model"
    
    # Test that configuration values are reasonable (not exact values)
    assert settings.api_timeout > 0
    assert settings.dcs_start_turn >= 1
    assert settings.hes_start_turn >= 1
    assert settings.hes_start_turn >= settings.dcs_start_turn  # HES should start after or same as DCS
    assert settings.api_retry_attempts > 0
    assert settings.api_retry_delay >= 0
    assert settings.output_directory is not None
    assert settings.max_concurrent_requests > 0
    assert settings.max_concurrent_requests <= 100  # Should not exceed reasonable limit
    
    # Test validation - may pass if env var is set
    is_valid, errors = settings.validate_setup()
    if not settings.has_api_key:
        assert not is_valid
        assert len(errors) > 0
        assert "OpenRouter API key" in errors[0]


def test_settings_override():
    """Test that settings can be overridden via constructor."""
    # Test overriding via constructor
    custom_settings = Settings(
        default_target_model="custom/model",
        api_timeout=600,
        dcs_start_turn=5,
        max_concurrent_requests=50
    )
    
    assert custom_settings.default_target_model == "custom/model"
    assert custom_settings.api_timeout == 600
    assert custom_settings.dcs_start_turn == 5
    assert custom_settings.max_concurrent_requests == 50
    # Other values should be reasonable
    assert custom_settings.hes_start_turn >= 1
    assert custom_settings.api_retry_attempts > 0


def test_settings_validation():
    """Test settings validation limits."""
    # Test max_concurrent_requests validation
    try:
        # This should work (within limits)
        valid_settings = Settings(max_concurrent_requests=50)
        assert valid_settings.max_concurrent_requests == 50
        
        # This should fail (exceeds max limit)
        invalid_settings = Settings(max_concurrent_requests=200)
        assert False, "Should have failed with value > 100"
        
    except Exception as e:
        # Expected to fail for invalid values
        assert str(settings.max_concurrent_requests) in str(e) or "100" in str(e) or "le" in str(e).lower()
    
    try:
        # This should fail (below minimum)
        invalid_settings = Settings(max_concurrent_requests=0)
        assert False, "Should have failed with value < 1"
        
    except Exception as e:
        # Expected to fail for invalid values
        assert "1" in str(e) or "ge" in str(e).lower()


@patch('psy_bench.core.api_client.requests.post')
def test_api_client(mock_post):
    """Test OpenRouter API client."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 10}
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    client = OpenRouterClient(api_key="test-key")
    
    messages = [ChatMessage(role="user", content="Test message")]
    response = client.chat("test-model", messages)
    
    assert response.content == "Test response"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 10
    
    # Verify the API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["model"] == "test-model"
    assert call_args[1]["json"]["messages"] == [{"role": "user", "content": "Test message"}]


def test_conversation_turn():
    """Test ConversationTurn model."""
    turn = ConversationTurn(
        user="Test user message",
        bot="Test bot response",
        turn_number=1
    )
    
    assert turn.user == "Test user message"
    assert turn.bot == "Test bot response"
    assert turn.turn_number == 1


def test_scoring_basic():
    """Test basic scoring functionality."""
    # Mock the API client
    mock_client = Mock()
    mock_client.get_judge_response.return_value = "Score: 1"
    
    scorer = DelusionConfirmationScorer(mock_client, "test-judge-model")
    
    # Test conversation history
    conversation = [
        ConversationTurn(user="Test user message", bot="Test bot response", turn_number=1)
    ]
    
    result = scorer.score(conversation)
    
    # Should call the judge model
    mock_client.get_judge_response.assert_called_once()
    
    # Should parse the score
    assert result.score == 1


def test_case_properties():
    """Test TestCase properties and methods."""
    case = get_case_by_name("Case 1.1: The Conduit [EXPLICIT]")
    
    assert case.is_explicit == True
    assert case.is_implicit == False
    
    # Test phase organization
    phases = case.phase_prompts
    assert len(phases) == 4  # 4 phases
    assert all(len(phase) == 3 for phase in phases)  # 3 prompts per phase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])