
"""
LLM adapter (Gemini-only) for the RTGS pipeline.

- Uses google-generativeai (import name: google.generativeai).
- Reads GEMINI_API_KEY and GEMINI_MODEL from .env.
- Tries multiple call shapes so it is tolerant to small SDK differences.
- On failure, prints a manual prompt (safe fallback) so pipeline can continue.
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Config from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Defaults
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))


def _manual_prompt_fallback(prompt: str, out_path: Optional[str] = None) -> str:
    header = "\n=== MANUAL LLM PROMPT (copy this into your LLM UI) ===\n"
    footer = "\n=== END PROMPT ===\n"
    manual_text = f"{header}{prompt}{footer}"
    print(manual_text)
    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(manual_text)
        except Exception:
            pass
    return "NO_AUTOMATIC_LLM_RESPONSE — prompt printed; paste model answer here."


def _wrap_system_prompt(user_prompt: str) -> str:
    system = "You are an evidence-focused analyst. Use ONLY the provided evidence."
    return f"{system}\n\n{user_prompt}"


def _extract_text_from_response(resp) -> str:
    try:
        if hasattr(resp, "text") and resp.text:
            return str(resp.text).strip()
    except Exception:
        pass
    try:
        out = getattr(resp, "output", None)
        if out and isinstance(out, (list, tuple)) and len(out) > 0:
            piece = out[0]
            try:
                return str(piece.content[0].text).strip()
            except Exception:
                return str(piece).strip()
    except Exception:
        pass
    try:
        cand = getattr(resp, "candidates", None)
        if cand and len(cand) > 0:
            c0 = cand[0]
            return (getattr(c0, "text", None) or getattr(c0, "content", None) or str(c0)).strip()
    except Exception:
        pass
    try:
        return str(resp).strip()
    except Exception:
        return ""


def call_gemini(prompt: str,
                model: Optional[str] = None,
                temperature: float = DEFAULT_TEMPERATURE,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set in environment. Falling back to manual prompt.")
        return _manual_prompt_fallback(_wrap_system_prompt(prompt))

    model = model or GEMINI_MODEL
    prompt_full = _wrap_system_prompt(prompt)

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        print("Failed to import google.generativeai:", repr(e))
        return _manual_prompt_fallback(prompt_full)

    try:
        if hasattr(genai, "configure"):
            try:
                genai.configure(api_key=GEMINI_API_KEY)
            except Exception:
                pass
    except Exception:
        pass

    resp = None
    last_err = None

    try:
        if hasattr(genai, "GenerativeModel"):
            gm = genai.GenerativeModel(model)
            try:
                resp = gm.generate_content(prompt_full)
            except TypeError:
                try:
                    resp = gm.generate_content([{"type": "text", "text": prompt_full}])
                except Exception:
                    try:
                        resp = gm.generate_text(prompt_full)
                    except Exception:
                        resp = None
    except Exception as e:
        last_err = e

    if resp is None:
        try:
            if hasattr(genai, "generate_text"):
                try:
                    resp = genai.generate_text(model=model, prompt=prompt_full)
                except TypeError:
                    resp = genai.generate_text(model=model, input=prompt_full)
            if resp is None and hasattr(genai, "generate"):
                resp = genai.generate(model=model, input=prompt_full)
        except Exception as e:
            last_err = e

    if resp is None:
        if last_err:
            print("Gemini generate attempt failed (last_err):", repr(last_err))
        else:
            print("Gemini generate attempt failed: no supported method found on the genai client.")
        print("Falling back to manual prompt.")
        return _manual_prompt_fallback(prompt_full)

    try:
        text = _extract_text_from_response(resp)
        if text:
            return text
        else:
            print("Gemini returned empty text; falling back to manual prompt.")
            return _manual_prompt_fallback(prompt_full)
    except Exception as e:
        print("Error extracting text from Gemini response:", repr(e))
        return _manual_prompt_fallback(prompt_full)


# Backwards-compatible public function
def call_llm(prompt: str,
             model: Optional[str] = None,
             temperature: float = DEFAULT_TEMPERATURE,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             provider: Optional[str] = None,
             **kwargs) -> str:
    """
    Public adapter that pipeline code calls.

    Accepts 'provider' (ignored, Gemini-only) for backwards compatibility.
    """
    if provider and provider.lower() != "gemini":
        print(f"Warning: requested provider={provider}, but this adapter is Gemini-only. Using Gemini anyway.")
    return call_gemini(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    test_prompt = "Sanity check: reply with 'agent-ok' only."
    print("Running quick Gemini connectivity check (no secrets shown).")
    out = call_llm(test_prompt, provider="gemini")
    print("Result:", out)
