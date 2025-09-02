"""Unified OpenRouter API client with sync and async support."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel, Field

from ..config import DEFAULT_JUDGE_MODEL, OPENROUTER_API_KEY


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatResponse(BaseModel):
    """Response from the API."""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model that generated the response")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")


class OpenRouterClient:
    """Unified client for OpenRouter API with sync and async support."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 1,
    ):
        """Initialize the client.
        
        Args:
            api_key: OpenRouter API key (uses settings if not provided)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def _prepare_request(self, model: str, messages: List[ChatMessage]) -> tuple[dict, dict]:
        """Prepare headers and payload for request.
        
        Args:
            model: Model to use
            messages: Conversation messages
            
        Returns:
            Tuple of (headers, payload)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Psy-Bench AI Safety Evaluation"
        }
        
        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages]
        }
        
        return headers, payload
    
    def _parse_response(self, response_data: dict, model: str) -> ChatResponse:
        """Parse API response into ChatResponse object.
        
        Args:
            response_data: Raw response from API
            model: Model name
            
        Returns:
            ChatResponse object
        """
        try:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content.strip(),
                model=model,
                usage=usage
            )
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {e}")
    
    def chat(self, model: str, messages: List[ChatMessage]) -> ChatResponse:
        """Send a synchronous chat request.
        
        Args:
            model: Model to use
            messages: Conversation messages
            
        Returns:
            ChatResponse from the model
        """
        headers, payload = self._prepare_request(model, messages)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_response(response.json(), model)
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                # Try to get detailed error response
                error_details = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    try:
                        error_details = f"{e} - Response: {e.response.text}"
                    except:
                        pass
                raise RuntimeError(f"API request failed after {self.max_retries} attempts: {error_details}")
    
    async def achat(self, model: str, messages: List[ChatMessage]) -> ChatResponse:
        """Send an asynchronous chat request.
        
        Args:
            model: Model to use
            messages: Conversation messages
            
        Returns:
            ChatResponse from the model
        """
        headers, payload = self._prepare_request(model, messages)
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.BASE_URL,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return self._parse_response(data, model)
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    # Try to get detailed error response
                    error_details = str(e)
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        try:
                            error_text = await e.response.text()
                            error_details = f"{e} - Response: {error_text}"
                        except:
                            pass
                    raise RuntimeError(f"API request failed after {self.max_retries} attempts: {error_details}")
    
    def judge(self, prompt: str, model: Optional[str] = None) -> str:
        """Get a judge response for scoring.
        
        Args:
            prompt: Scoring prompt
            model: Judge model to use (uses default if not provided)
            
        Returns:
            Judge response as string
        """
        judge_model = model or DEFAULT_JUDGE_MODEL
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.chat(judge_model, messages)
        return response.content
    
    async def ajudge(self, prompt: str, model: Optional[str] = None) -> str:
        """Get an async judge response for scoring.
        
        Args:
            prompt: Scoring prompt
            model: Judge model to use (uses default if not provided)
            
        Returns:
            Judge response as string
        """
        judge_model = model or DEFAULT_JUDGE_MODEL
        messages = [ChatMessage(role="user", content=prompt)]
        response = await self.achat(judge_model, messages)
        return response.content
