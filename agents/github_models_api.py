#!/usr/bin/env python3
# Modified: 2026-02-06T22:00:00Z | Author: Claude | Change: GitHub Models API for dashboard integration
"""
SLATE GitHub Models API
========================
Provides HTTP endpoints for GitHub Models integration in the SLATE Dashboard.

Endpoints:
    GET  /api/models/status      → GitHub Models availability status
    GET  /api/models/list        → List available models
    POST /api/models/chat        → Send a chat completion
    POST /api/models/embed       → Generate embeddings
    GET  /api/models/rate-limits → Get rate limit info
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def add_github_models_endpoints(app):
    """Register GitHub Models API endpoints on a FastAPI app instance."""
    from fastapi import BackgroundTasks, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        prompt: str
        system: Optional[str] = None
        model: str = "gpt-4o-mini"
        temperature: float = 0.7
        max_tokens: int = 2048
        stream: bool = False

    class EmbedRequest(BaseModel):
        texts: List[str]
        model: str = "text-embedding-3-small"

    @app.get("/api/models/status")
    async def models_status():
        """Return GitHub Models availability status."""
        try:
            from slate.slate_github_models import check_availability
            status = check_availability()
            return JSONResponse(content=status)
        except ImportError:
            return JSONResponse(content={
                "available": False,
                "error": "slate_github_models module not found",
            })
        except Exception as e:
            return JSONResponse(content={
                "available": False,
                "error": str(e),
            }, status_code=500)

    @app.get("/api/models/list")
    async def models_list():
        """Return list of available models."""
        try:
            from slate.slate_github_models import AVAILABLE_MODELS, RATE_LIMITS
            models = []
            for name, info in AVAILABLE_MODELS.items():
                tier = info.get("tier", "low")
                limits = RATE_LIMITS.get(tier, {})
                models.append({
                    "name": name,
                    "provider": info.get("provider"),
                    "type": info.get("type"),
                    "tier": tier,
                    "rpm": limits.get("rpm"),
                    "rpd": limits.get("rpd"),
                })
            return JSONResponse(content={"models": models, "count": len(models)})
        except ImportError:
            return JSONResponse(content={"models": [], "error": "Module not found"})
        except Exception as e:
            return JSONResponse(content={"models": [], "error": str(e)}, status_code=500)

    @app.post("/api/models/chat")
    async def models_chat(request: ChatRequest):
        """Send a chat completion request."""
        try:
            from slate.slate_github_models import GitHubModelsClient

            client = GitHubModelsClient(model=request.model)

            if request.stream:
                # Return streaming response
                async def generate():
                    try:
                        for chunk in client.chat(
                            prompt=request.prompt,
                            system=request.system,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            stream=True,
                        ):
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )

            response = client.chat(
                prompt=request.prompt,
                system=request.system,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            return JSONResponse(content={
                "content": response.content,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            })

        except ImportError:
            return JSONResponse(content={
                "error": "azure-ai-inference not installed. Run: pip install azure-ai-inference"
            }, status_code=500)
        except ValueError as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.post("/api/models/embed")
    async def models_embed(request: EmbedRequest):
        """Generate embeddings."""
        try:
            from slate.slate_github_models import GitHubModelsClient

            client = GitHubModelsClient()
            embeddings = client.embed(request.texts, model=request.model)

            return JSONResponse(content={
                "embeddings": embeddings,
                "count": len(embeddings),
                "model": request.model,
                "dimensions": len(embeddings[0]) if embeddings else 0,
            })

        except ImportError:
            return JSONResponse(content={
                "error": "azure-ai-inference not installed"
            }, status_code=500)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.get("/api/models/rate-limits")
    async def models_rate_limits():
        """Return rate limit information."""
        try:
            from slate.slate_github_models import RATE_LIMITS
            return JSONResponse(content={"rate_limits": RATE_LIMITS})
        except ImportError:
            return JSONResponse(content={
                "rate_limits": {
                    "low": {"rpm": 15, "rpd": 150},
                    "high": {"rpm": 10, "rpd": 50},
                    "embedding": {"rpm": 15, "rpd": 150},
                }
            })


# ─── Standalone test server ─────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        print("[ERROR] FastAPI/uvicorn not installed")
        sys.exit(1)

    app = FastAPI(title="SLATE GitHub Models API")
    add_github_models_endpoints(app)

    @app.get("/")
    async def root():
        return {"message": "SLATE GitHub Models API", "endpoints": [
            "/api/models/status",
            "/api/models/list",
            "/api/models/chat",
            "/api/models/embed",
            "/api/models/rate-limits",
        ]}

    print("[SLATE] Starting GitHub Models API on http://127.0.0.1:8082")
    uvicorn.run(app, host="127.0.0.1", port=8082, log_level="info")
