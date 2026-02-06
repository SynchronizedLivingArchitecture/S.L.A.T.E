#!/usr/bin/env python3
# Modified: 2026-02-06T22:00:00Z | Author: Claude | Change: GitHub Models integration for SLATE
"""
SLATE GitHub Models Integration
================================
Provides access to GitHub Models (AI models via GitHub Marketplace) for
local inference through the self-hosted runner.

Models available at: https://github.com/marketplace/models

Authentication:
    Requires a GitHub PAT with `models:read` permission.
    Set via GITHUB_TOKEN environment variable or pass directly.

Usage:
    from slate.slate_github_models import GitHubModelsClient

    client = GitHubModelsClient()
    response = client.chat("What is SLATE?")
    print(response)

CLI:
    python slate/slate_github_models.py --list-models
    python slate/slate_github_models.py --chat "Hello"
    python slate/slate_github_models.py --model gpt-4o --chat "Explain quantum computing"
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# GitHub Models endpoint
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Available models on GitHub Models (as of 2026)
# Full list at: https://github.com/marketplace/models
AVAILABLE_MODELS = {
    # OpenAI models
    "gpt-4o": {"provider": "OpenAI", "type": "chat", "tier": "high"},
    "gpt-4o-mini": {"provider": "OpenAI", "type": "chat", "tier": "low"},
    "gpt-4-turbo": {"provider": "OpenAI", "type": "chat", "tier": "high"},
    "o1-preview": {"provider": "OpenAI", "type": "chat", "tier": "high"},
    "o1-mini": {"provider": "OpenAI", "type": "chat", "tier": "low"},

    # Meta Llama models
    "meta-llama-3.1-405b-instruct": {"provider": "Meta", "type": "chat", "tier": "high"},
    "meta-llama-3.1-70b-instruct": {"provider": "Meta", "type": "chat", "tier": "high"},
    "meta-llama-3.1-8b-instruct": {"provider": "Meta", "type": "chat", "tier": "low"},
    "llama-3.2-90b-vision-instruct": {"provider": "Meta", "type": "multimodal", "tier": "high"},
    "llama-3.2-11b-vision-instruct": {"provider": "Meta", "type": "multimodal", "tier": "low"},

    # Mistral models
    "mistral-large": {"provider": "Mistral AI", "type": "chat", "tier": "high"},
    "mistral-small": {"provider": "Mistral AI", "type": "chat", "tier": "low"},
    "mistral-nemo": {"provider": "Mistral AI", "type": "chat", "tier": "low"},

    # Cohere models
    "cohere-command-r-plus": {"provider": "Cohere", "type": "chat", "tier": "high"},
    "cohere-command-r": {"provider": "Cohere", "type": "chat", "tier": "low"},

    # AI21 models
    "ai21-jamba-1.5-large": {"provider": "AI21 Labs", "type": "chat", "tier": "high"},
    "ai21-jamba-1.5-mini": {"provider": "AI21 Labs", "type": "chat", "tier": "low"},

    # Embedding models
    "text-embedding-3-large": {"provider": "OpenAI", "type": "embedding", "tier": "embedding"},
    "text-embedding-3-small": {"provider": "OpenAI", "type": "embedding", "tier": "embedding"},
    "cohere-embed-v3-english": {"provider": "Cohere", "type": "embedding", "tier": "embedding"},
    "cohere-embed-v3-multilingual": {"provider": "Cohere", "type": "embedding", "tier": "embedding"},
}

# Rate limits by tier (requests per minute / requests per day)
RATE_LIMITS = {
    "low": {"rpm": 15, "rpd": 150, "tokens_in": 8000, "tokens_out": 4000},
    "high": {"rpm": 10, "rpd": 50, "tokens_in": 8000, "tokens_out": 4000},
    "embedding": {"rpm": 15, "rpd": 150, "tokens_in": 64000, "tokens_out": 0},
}


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Optional[Dict] = None


class GitHubModelsClient:
    """
    Client for GitHub Models API.

    Uses the Azure AI Inference SDK to access models available through
    GitHub's model marketplace.

    Example:
        client = GitHubModelsClient()
        response = client.chat("What is Python?")
        print(response.content)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = "gpt-4o-mini",
        endpoint: str = GITHUB_MODELS_ENDPOINT,
    ):
        """
        Initialize the GitHub Models client.

        Args:
            token: GitHub PAT with models:read permission. If not provided,
                   uses GITHUB_TOKEN environment variable.
            model: Default model to use.
            endpoint: GitHub Models API endpoint.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN env var or pass token parameter. "
                "Token needs 'models:read' permission."
            )

        self.model = model
        self.endpoint = endpoint
        self._client = None
        self._embeddings_client = None

    def _get_chat_client(self):
        """Get or create the chat completions client."""
        if self._client is None:
            try:
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError:
                raise ImportError(
                    "azure-ai-inference package required. "
                    "Install with: pip install azure-ai-inference"
                )

            self._client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
            )
        return self._client

    def _get_embeddings_client(self):
        """Get or create the embeddings client."""
        if self._embeddings_client is None:
            try:
                from azure.ai.inference import EmbeddingsClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError:
                raise ImportError(
                    "azure-ai-inference package required. "
                    "Install with: pip install azure-ai-inference"
                )

            self._embeddings_client = EmbeddingsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
            )
        return self._embeddings_client

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Union[ChatResponse, Generator[str, None, None]]:
        """
        Send a chat completion request.

        Args:
            prompt: The user's message.
            system: Optional system message.
            model: Model to use (overrides default).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            stream: If True, yields response chunks.

        Returns:
            ChatResponse or generator of string chunks if streaming.
        """
        try:
            from azure.ai.inference.models import SystemMessage, UserMessage
        except ImportError:
            raise ImportError("azure-ai-inference package required")

        client = self._get_chat_client()
        use_model = model or self.model

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(UserMessage(content=prompt))

        if stream:
            return self._chat_stream(client, messages, use_model, temperature, max_tokens)

        response = client.complete(
            model=use_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw_response=response.as_dict() if hasattr(response, "as_dict") else None,
        )

    def _chat_stream(
        self, client, messages, model: str, temperature: float, max_tokens: int
    ) -> Generator[str, None, None]:
        """Stream chat completion responses."""
        response = client.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for update in response:
            if update.choices and update.choices[0].delta:
                content = update.choices[0].delta.content
                if content:
                    yield content

    def chat_with_history(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """
        Send a chat completion with message history.

        Args:
            messages: List of ChatMessage objects.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            ChatResponse object.
        """
        try:
            from azure.ai.inference.models import (
                SystemMessage, UserMessage, AssistantMessage
            )
        except ImportError:
            raise ImportError("azure-ai-inference package required")

        client = self._get_chat_client()
        use_model = model or self.model

        msg_objects = []
        for msg in messages:
            if msg.role == "system":
                msg_objects.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                msg_objects.append(UserMessage(content=msg.content))
            elif msg.role == "assistant":
                msg_objects.append(AssistantMessage(content=msg.content))

        response = client.complete(
            model=use_model,
            messages=msg_objects,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small",
    ) -> List[List[float]]:
        """
        Generate embeddings for text.

        Args:
            texts: Text or list of texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors.
        """
        client = self._get_embeddings_client()

        if isinstance(texts, str):
            texts = [texts]

        response = client.embed(
            model=model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Return available models and their info."""
        return AVAILABLE_MODELS.copy()

    def get_rate_limits(self, model: Optional[str] = None) -> Dict[str, int]:
        """Get rate limits for a model tier."""
        if model:
            tier = AVAILABLE_MODELS.get(model, {}).get("tier", "low")
        else:
            tier = "low"
        return RATE_LIMITS.get(tier, RATE_LIMITS["low"])


# ─── Integration with SLATE Unified Backend ─────────────────────────────────────

def create_slate_backend() -> Dict[str, Any]:
    """
    Create a SLATE-compatible backend configuration for GitHub Models.

    Returns:
        Backend configuration dict for unified_ai_backend.py
    """
    return {
        "name": "github_models",
        "display_name": "GitHub Models",
        "type": "cloud",
        "endpoint": GITHUB_MODELS_ENDPOINT,
        "auth_type": "token",
        "auth_env": "GITHUB_TOKEN",
        "models": list(AVAILABLE_MODELS.keys()),
        "default_model": "gpt-4o-mini",
        "capabilities": ["chat", "embedding"],
        "rate_limited": True,
        "cost": "free_tier",  # Free for GitHub users with limits
    }


def check_availability() -> Dict[str, Any]:
    """
    Check if GitHub Models is available.

    Returns:
        Status dict with availability info.
    """
    result = {
        "available": False,
        "token_set": False,
        "sdk_installed": False,
        "test_passed": False,
        "error": None,
    }

    # Check token
    token = os.environ.get("GITHUB_TOKEN")
    result["token_set"] = bool(token)

    # Check SDK
    try:
        from azure.ai.inference import ChatCompletionsClient
        result["sdk_installed"] = True
    except ImportError:
        result["error"] = "azure-ai-inference not installed"
        return result

    if not token:
        result["error"] = "GITHUB_TOKEN not set"
        return result

    # Test connection
    try:
        client = GitHubModelsClient(token=token, model="gpt-4o-mini")
        response = client.chat("Say 'test' and nothing else.", max_tokens=10)
        if response and response.content:
            result["test_passed"] = True
            result["available"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


# ─── GitHub Actions Integration ─────────────────────────────────────────────────

def generate_workflow_snippet() -> str:
    """
    Generate a GitHub Actions workflow snippet for using GitHub Models.

    The runner automatically has access to GITHUB_TOKEN, so no extra
    setup is needed when running on the self-hosted runner.
    """
    return """# GitHub Models in GitHub Actions
# The runner automatically has GITHUB_TOKEN available

- name: Use GitHub Models
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    python -c "
    from slate.slate_github_models import GitHubModelsClient
    client = GitHubModelsClient()
    response = client.chat('Analyze this code...', model='gpt-4o-mini')
    print(response.content)
    "

- name: Code Review with AI
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    python slate/slate_github_models.py --model gpt-4o --chat "Review this PR: ${{ github.event.pull_request.body }}"
"""


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SLATE GitHub Models Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python slate_github_models.py --list-models

  # Simple chat
  python slate_github_models.py --chat "What is Python?"

  # Chat with specific model
  python slate_github_models.py --model gpt-4o --chat "Explain async/await"

  # Streaming output
  python slate_github_models.py --chat "Write a poem" --stream

  # Check availability
  python slate_github_models.py --check

  # Show workflow snippet
  python slate_github_models.py --workflow
"""
    )

    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--chat", type=str, help="Chat message to send")
    parser.add_argument("--system", type=str, help="System message")
    parser.add_argument("--stream", action="store_true", help="Stream response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens")
    parser.add_argument("--check", action="store_true", help="Check availability")
    parser.add_argument("--workflow", action="store_true", help="Show workflow snippet")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    args = parser.parse_args()

    if args.list_models:
        print("\n[GitHub Models - Available Models]")
        print("=" * 60)
        for name, info in AVAILABLE_MODELS.items():
            limits = RATE_LIMITS.get(info["tier"], {})
            print(f"  {name}")
            print(f"    Provider: {info['provider']}")
            print(f"    Type: {info['type']}")
            print(f"    Tier: {info['tier']} ({limits.get('rpm', '?')} rpm, {limits.get('rpd', '?')} rpd)")
            print()
        print("Full list: https://github.com/marketplace/models")
        return

    if args.check:
        status = check_availability()
        if args.json_output:
            print(json.dumps(status, indent=2))
        else:
            print("\n[GitHub Models - Availability Check]")
            print("=" * 60)
            print(f"  Token Set:      {'YES' if status['token_set'] else 'NO'}")
            print(f"  SDK Installed:  {'YES' if status['sdk_installed'] else 'NO'}")
            print(f"  Test Passed:    {'YES' if status['test_passed'] else 'NO'}")
            print(f"  Available:      {'YES' if status['available'] else 'NO'}")
            if status['error']:
                print(f"  Error:          {status['error']}")
            print()
            if not status['sdk_installed']:
                print("  Install SDK: pip install azure-ai-inference")
            if not status['token_set']:
                print("  Set token: export GITHUB_TOKEN=your_pat_here")
        return

    if args.workflow:
        print(generate_workflow_snippet())
        return

    if args.chat:
        try:
            client = GitHubModelsClient(model=args.model)

            if args.stream:
                print(f"\n[{args.model}] ", end="", flush=True)
                for chunk in client.chat(
                    args.chat,
                    system=args.system,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True,
                ):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = client.chat(
                    args.chat,
                    system=args.system,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if args.json_output:
                    print(json.dumps({
                        "content": response.content,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    }, indent=2))
                else:
                    print(f"\n[{response.model}]")
                    print("-" * 40)
                    print(response.content)
                    print("-" * 40)
                    print(f"Tokens: {response.usage['total_tokens']} "
                          f"(prompt: {response.usage['prompt_tokens']}, "
                          f"completion: {response.usage['completion_tokens']})")

        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
