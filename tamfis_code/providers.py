"""Provider integrations for TAMFIS-CODE - HF, NVIDIA NIM, OpenRouter, and Ollama"""

import os
import json
import subprocess
from typing import Optional, Dict, Any, AsyncIterator, List
from dataclasses import dataclass, field
from enum import Enum
import httpx
from openai import AsyncOpenAI


class ProviderType(Enum):
    HF = "hf"
    NVIDIA = "nvidia"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LOCAL = "local"
    AUTO = "auto"


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    models: List[str] = field(default_factory=list)
    weight: int = 1
    reasoning_supported: bool = False
    vision_supported: bool = False


class ProviderManager:
    """Manages multiple AI providers with vision support"""
    
    PROVIDERS = {
        # NOTE: provider catalogs drift as vendors ship/retire models --
        # these lists are a reasonable, current-as-of-writing starting point
        # for local/offline mode, not a live catalog feed. Review periodically.
        ProviderType.HF: ProviderConfig(
            name="Hugging Face",
            base_url="https://router.huggingface.co/v1",
            api_key_env="HF_TOKEN",
            default_model="meta-llama/Llama-3.2-3B-Instruct",
            models=[
                # Text models
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "Qwen/Qwen2.5-7B-Instruct",
                # Vision models
                "microsoft/Phi-3.5-vision-instruct",
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "Qwen/Qwen2-VL-7B-Instruct",
            ],
            weight=2,
            vision_supported=True
        ),
        ProviderType.NVIDIA: ProviderConfig(
            name="NVIDIA NIM (API Catalog)",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            default_model="nvidia/nemotron-3-super-120b-a12b",
            models=[
                # Text models
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "moonshotai/kimi-k2.6",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct",
                "mistralai/mistral-large-2-123b",
                "google/gemma-2-27b-it",
                "microsoft/phi-3-medium-128k-instruct",
            ],
            weight=3,
            reasoning_supported=True,
            vision_supported=False
        ),
        ProviderType.OPENROUTER: ProviderConfig(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_model="openai/gpt-4o-mini",
            models=[
                # Text models
                "openai/gpt-4o-mini",
                "google/gemini-2.5-flash",
                "meta-llama/llama-3.3-70b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
                # Vision models
                "google/gemini-2.5-flash",
                "openai/gpt-4o-mini",
            ],
            weight=1,
            vision_supported=True
        ),
        ProviderType.OLLAMA: ProviderConfig(
            name="Ollama (Local)",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key_env="OLLAMA_API_KEY",
            default_model="llama3.2:3b",
            models=[
                "llama3.2:3b",
                "llama3.2:1b",
                "phi3:mini",
                "qwen2.5:3b",
                "qwen2.5:7b",
                "codellama:7b",
                "codellama:13b",
                "mistral:7b",
                "mixtral:8x7b",
                "tamgpt-fast-1:latest",
                "tamgpt-r1-elite:latest",
                "tamgpt-vision-pro:latest",
                "deepseek-coder:6.7b",
                "gemma2:9b",
                "nomic-embed-text:latest",
                # Vision-capable (if available)
                "llava:7b",
                "llava:13b",
                "bakllava:7b",
            ],
            weight=4,
            reasoning_supported=True,
            vision_supported=True
        ),
    }
    
    def __init__(self):
        self.clients: Dict[ProviderType, AsyncOpenAI] = {}
        self.config = self._load_config()
        self._init_clients()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load provider configuration from environment"""
        config = {}
        for provider in ProviderType:
            if provider == ProviderType.AUTO:
                continue
            env_var = f"TAMFIS_PROVIDER_{provider.value.upper()}_ENABLED"
            config[provider.value] = os.environ.get(env_var, "true").lower() == "true"
        return config
    
    def _get_api_key(self, provider_type: ProviderType) -> Optional[str]:
        """Get API key for a provider from environment"""
        config = self.PROVIDERS.get(provider_type)
        if not config:
            return None
        
        # Ollama doesn't need a key
        if provider_type == ProviderType.OLLAMA:
            return "ollama"
        
        key = os.environ.get(config.api_key_env, "")
        return key if key else None
    
    def _has_valid_api_key(self, provider_type: ProviderType) -> bool:
        """Check if a provider has a valid API key set"""
        if provider_type == ProviderType.AUTO:
            return False
        
        # Ollama always available if the service is running
        if provider_type == ProviderType.OLLAMA:
            return self._check_ollama_available()
        
        config = self.PROVIDERS.get(provider_type)
        if not config:
            return False
        
        key = os.environ.get(config.api_key_env, "")
        if not key:
            return False
        
        # Check if it's a placeholder
        placeholder_patterns = ["your_", "_key", "_token", "YOUR_", "_API_KEY"]
        if any(p in key for p in placeholder_patterns):
            return False
        
        # Valid key must be at least 8 characters
        if len(key) < 8:
            return False
        
        return True
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama service is running and accessible"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0 and "NAME" in result.stdout
        except:
            return False
    
    def _get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
        except:
            pass
        return []
    
    def _init_clients(self):
        """Initialize OpenAI-compatible clients for each provider"""
        for provider_type, config in self.PROVIDERS.items():
            if provider_type == ProviderType.AUTO:
                continue
            if not self.config.get(provider_type.value, True):
                continue
            
            # Skip if no valid API key (except Ollama)
            if not self._has_valid_api_key(provider_type) and provider_type != ProviderType.OLLAMA:
                continue
            
            api_key = self._get_api_key(provider_type)
            
            try:
                self.clients[provider_type] = AsyncOpenAI(
                    base_url=config.base_url,
                    api_key=api_key or "dummy",
                    timeout=120.0,
                    max_retries=2,
                )
            except Exception as e:
                print(f"Failed to initialize {config.name}: {e}")
    
    def get_client(self, provider: ProviderType) -> Optional[AsyncOpenAI]:
        """Get a client for a specific provider"""
        if provider == ProviderType.AUTO:
            provider = self._select_best_provider()
        return self.clients.get(provider)
    
    def _select_best_provider(self) -> ProviderType:
        """Select the best available provider based on weight and availability"""
        available = []
        for p, client in self.clients.items():
            if client and self._has_valid_api_key(p):
                available.append((p, self.PROVIDERS[p].weight))
        
        if not available:
            if self._check_ollama_available():
                return ProviderType.OLLAMA
            return ProviderType.NVIDIA
        
        available.sort(key=lambda x: x[1], reverse=True)
        return available[0][0]
    
    def list_available_providers(self) -> List[Dict[str, Any]]:
        """List all available providers with their status"""
        result = []
        ollama_models = self._get_ollama_models() if self._check_ollama_available() else []
        
        for provider_type, client in self.clients.items():
            config = self.PROVIDERS.get(provider_type)
            if config:
                valid_key = self._has_valid_api_key(provider_type)
                available = client is not None and valid_key
                
                if provider_type == ProviderType.OLLAMA:
                    available = self._check_ollama_available()
                    valid_key = available
                
                result.append({
                    "name": config.name,
                    "type": provider_type.value,
                    "available": available,
                    "default_model": config.default_model,
                    "models": ollama_models if provider_type == ProviderType.OLLAMA else config.models,
                    "weight": config.weight,
                    "api_key_set": valid_key,
                    "reasoning_supported": config.reasoning_supported,
                    "vision_supported": config.vision_supported,
                    "key_preview": self._get_api_key(provider_type)[:8] + "..." if self._get_api_key(provider_type) and provider_type != ProviderType.OLLAMA else "local"
                })
        return result
    
    async def chat_completion(
        self,
        provider: ProviderType,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = True,
        temperature: float = 0.10,
        max_tokens: int = 16384,
        reasoning_effort: Optional[str] = "high",
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a chat completion from a provider"""
        client = self.get_client(provider)
        if not client:
            raise ValueError(f"Provider {provider} not available")
        
        config = self.PROVIDERS.get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")
        
        model = model or config.default_model
        
        # For Ollama, use a valid model name
        if provider == ProviderType.OLLAMA:
            ollama_models = self._get_ollama_models()
            if model not in ollama_models and ollama_models:
                model = ollama_models[0]
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if stream:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                if response.choices and response.choices[0].message.content:
                    yield response.choices[0].message.content
                    
        except Exception as e:
            print(f"[{config.name} Error] {e}")
            fallback_chain = [ProviderType.OLLAMA, ProviderType.NVIDIA, ProviderType.HF]
            for fallback in fallback_chain:
                if fallback != provider and self.get_client(fallback):
                    try:
                        async for chunk in self.chat_completion(
                            fallback, messages, model, stream, temperature, max_tokens, **kwargs
                        ):
                            yield chunk
                        return
                    except:
                        continue
            raise
    
    async def chat_completion_sync(
        self,
        provider: ProviderType,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.10,
        max_tokens: int = 16384,
        **kwargs
    ) -> str:
        """Get a single chat completion (non-streaming)"""
        result = []
        async for chunk in self.chat_completion(
            provider, messages, model, False, temperature, max_tokens, **kwargs
        ):
            result.append(chunk)
        return ''.join(result)


# Convenience functions
async def chat_with_hf(messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(ProviderType.HF, messages, **kwargs):
        yield chunk


async def chat_with_nvidia(messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(ProviderType.NVIDIA, messages, **kwargs):
        yield chunk


async def chat_with_openrouter(messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(ProviderType.OPENROUTER, messages, **kwargs):
        yield chunk


async def chat_with_ollama(messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
    manager = ProviderManager()
    async for chunk in manager.chat_completion(ProviderType.OLLAMA, messages, **kwargs):
        yield chunk


def get_provider_status() -> Dict[str, Any]:
    manager = ProviderManager()
    return {
        "available": manager.list_available_providers(),
        "default": manager._select_best_provider().value if manager.clients else "none",
        "config": {
            p.value: {
                "enabled": manager.config.get(p.value, True),
                "api_key_set": manager._has_valid_api_key(p),
                "key_preview": manager._get_api_key(p)[:8] + "..." if manager._get_api_key(p) else "Not set"
            }
            for p in ProviderType
            if p != ProviderType.AUTO and p in manager.PROVIDERS
        }
    }
