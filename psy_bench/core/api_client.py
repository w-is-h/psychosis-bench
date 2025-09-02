"""OpenRouter API client for interacting with various AI models."""

import time
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

from ..config.settings import settings


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class ChatMessage(BaseModel):
    """Represents a chat message."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Represents a response from the chat API."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, uses settings.openrouter_api_key
        """
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = settings.api_timeout
        self.retry_attempts = settings.api_retry_attempts
        self.retry_delay = settings.api_retry_delay
    
    def _make_request(self, model: str, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Make a request to the OpenRouter API with retry logic.
        
        Args:
            model: The model name to use
            messages: List of chat messages
            
        Returns:
            Dict containing the API response
            
        Raises:
            APIError: If the request fails after all retry attempts
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "Psy-Bench AI Safety Evaluation",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
        
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                
        raise APIError(f"API request failed after {self.retry_attempts} attempts: {last_exception}")
    
    def chat(self, model: str, messages: List[ChatMessage]) -> ChatResponse:
        """Send a chat request to the specified model.
        
        Args:
            model: The model name to use
            messages: List of chat messages
            
        Returns:
            ChatResponse containing the model's response
        """
        response_data = self._make_request(model, messages)
        
        try:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content.strip(),
                model=model,
                usage=usage
            )
        except (KeyError, IndexError) as e:
            raise APIError(f"Invalid response format from API: {e}")
    
    def get_judge_response(self, judge_model: str, prompt: str) -> str:
        """Get a response from a judge model for scoring purposes.
        
        Args:
            judge_model: The model to use for judging
            prompt: The prompt to send to the judge model
            
        Returns:
            The judge model's response as a string
        """
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.chat(judge_model, messages)
        return response.content