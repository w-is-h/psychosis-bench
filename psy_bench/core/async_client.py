"""Async OpenRouter API client for concurrent requests."""

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel

from ..config.settings import settings
from .api_client import ChatMessage, ChatResponse, APIError


class AsyncOpenRouterClient:
    """Async client for concurrent OpenRouter API requests."""
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent: Optional[int] = None):
        """Initialize the async OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, uses settings.openrouter_api_key
            max_concurrent: Maximum concurrent requests. If None, uses settings.max_concurrent_requests
        """
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = settings.api_timeout
        self.retry_attempts = settings.api_retry_attempts
        self.retry_delay = settings.api_retry_delay
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests
        
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def _make_request(
        self, 
        session: aiohttp.ClientSession,
        model: str, 
        messages: List[ChatMessage]
    ) -> Dict[str, Any]:
        """Make an async request to the OpenRouter API with retry logic."""
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
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                
        raise APIError(f"API request failed after {self.retry_attempts} attempts: {last_exception}")
    
    async def chat(
        self, 
        session: aiohttp.ClientSession,
        model: str, 
        messages: List[ChatMessage]
    ) -> ChatResponse:
        """Send an async chat request to the specified model."""
        async with self.semaphore:  # Limit concurrent requests
            response_data = await self._make_request(session, model, messages)
            
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
    
    async def batch_chat(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[ChatResponse]:
        """Send multiple chat requests concurrently.
        
        Args:
            requests: List of request dictionaries with 'model' and 'messages' keys
            
        Returns:
            List of ChatResponse objects in the same order as requests
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for request in requests:
                model = request['model']
                messages = request['messages']
                task = self.chat(session, model, messages)
                tasks.append(task)
            
            # Execute all requests concurrently
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_judge_response(
        self, 
        session: aiohttp.ClientSession,
        judge_model: str, 
        prompt: str
    ) -> str:
        """Get an async response from a judge model for scoring purposes."""
        messages = [ChatMessage(role="user", content=prompt)]
        response = await self.chat(session, judge_model, messages)
        return response.content


class AsyncBatchProcessor:
    """Process multiple experiments concurrently."""
    
    def __init__(self, client: AsyncOpenRouterClient):
        self.client = client
    
    async def process_conversation_batch(
        self,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple conversations concurrently.
        
        Args:
            conversations: List of conversation dictionaries with experiment data
            
        Returns:
            List of processed conversation results
        """
        # Prepare batch requests for target models
        target_requests = []
        for conv in conversations:
            target_requests.append({
                'model': conv['target_model'],
                'messages': conv['messages'],
                'conversation_id': conv['id']
            })
        
        # Execute target model requests concurrently
        async with aiohttp.ClientSession() as session:
            target_tasks = []
            
            for request in target_requests:
                task = self.client.chat(
                    session,
                    request['model'], 
                    request['messages']
                )
                target_tasks.append(task)
            
            target_responses = await asyncio.gather(*target_tasks, return_exceptions=True)
        
        # Process responses and prepare results
        results = []
        for i, response in enumerate(target_responses):
            if isinstance(response, Exception):
                results.append({
                    'id': conversations[i]['id'],
                    'error': str(response),
                    'response': None
                })
            else:
                results.append({
                    'id': conversations[i]['id'],
                    'response': response,
                    'error': None
                })
        
        return results