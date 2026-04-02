"""TurboQuant Pipe — Open WebUI-style inlet/outlet middleware.

Inlet:  injects TurboQuant metadata into chat requests.
Outlet: appends KV cache stats to assistant responses (if available).

Usage (from app.py or a future plugin loader):
    pipe = TurboQuantPipe()
    messages = pipe.inlet(messages, cache_type="turbo4")
    # ... proxy to llama.cpp ...
    result = pipe.outlet(result, meta={"cache_type": "turbo4"})
"""

from __future__ import annotations

import re
from typing import Optional

CACHE_TYPE_DESCRIPTIONS = {
    "turbo4": "TurboQuant 4-bit (3-bit PolarQuant + 1-bit QJL)",
    "turbo3": "TurboQuant 3-bit (2-bit PolarQuant + 1-bit QJL)",
    "q8_0":   "Standard 8-bit quantized KV cache",
    "q4_0":   "Standard 4-bit quantized KV cache",
    "f16":    "Full fp16 KV cache (no compression)",
}


class TurboQuantPipe:
    """Inlet/outlet pipe for TurboQuant-specific request enrichment."""

    def inlet(
        self,
        messages: list[dict],
        cache_type: str = "turbo4",
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Pre-process: inject system prompt with TurboQuant context.

        Args:
            messages: OpenAI-format message list.
            cache_type: Active KV cache type on the server.
            system_prompt: Base system prompt (from user config).

        Returns:
            Modified message list with system message prepended/updated.
        """
        base = system_prompt or "You are a helpful assistant."
        tq_note = (
            f"\n\n[KV Cache: {cache_type} — {CACHE_TYPE_DESCRIPTIONS.get(cache_type, cache_type)}]"
        )
        full_system = base + tq_note

        if messages and messages[0]["role"] == "system":
            # Replace existing system message
            out = [{"role": "system", "content": full_system}] + messages[1:]
        else:
            out = [{"role": "system", "content": full_system}] + messages

        return out

    def outlet(
        self,
        response_text: str,
        meta: Optional[dict] = None,
        show_stats: bool = False,
    ) -> str:
        """Post-process: optionally annotate response with cache stats.

        Args:
            response_text: Full assistant response text.
            meta: Optional metadata dict (cache_type, tokens, etc.).
            show_stats: Whether to append stats footer.

        Returns:
            (Possibly annotated) response text.
        """
        if not show_stats or meta is None:
            return response_text

        cache_type = meta.get("cache_type", "unknown")
        tokens = meta.get("tokens_generated", "?")
        tps = meta.get("tokens_per_second", "?")

        footer = (
            f"\n\n---\n*Cache: `{cache_type}` · "
            f"Tokens: {tokens} · Speed: {tps} tok/s*"
        )
        return response_text + footer
