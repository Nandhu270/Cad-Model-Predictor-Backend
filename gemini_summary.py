from fastapi import APIRouter, HTTPException
from typing import Any, Dict
import json
import traceback
import os
import logging
import asyncio

logger = logging.getLogger("uvicorn.error")

try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

router = APIRouter(prefix="/ai", tags=["ai"])

PROMPT_TEMPLATE = """
You are an expert piping/instrument QA assistant.
Input: a JSON array named "instruments" where each item is an object containing at least:
  - tag (string)
  - type (string)
  - measured: { upstream_m: number, downstream_m: number }
  - pass_fail: { upstream: bool, downstream: bool }
  - orientation: { tilt_deg: number, vertical_pass: bool }
  - suggestions: array of strings (optional)

Return STRICT JSON with keys:
{
  "total_instruments": int,
  "by_type": [{ "type": string, "count": int }],
  "failures": [
    {
      "tag": string,
      "type": string,
      "issues": [string,...],
      "details": { ... }
    }
  ],
  "summary_recommendations": [string],
  "confidence": "low|medium|high"
}
Output ONLY JSON.
"""

def _make_brief(ins):
    return {
        "tag": ins.get("tag") or ins.get("Tag"),
        "type": ins.get("type") or ins.get("Name") or ins.get("ObjectType"),
        "measured": ins.get("measured"),
        "pass_fail": ins.get("pass_fail"),
        "orientation": ins.get("orientation"),
        "suggestions": ins.get("suggestions", [])
    }

def _simple_local_summary(instruments):
    total = len(instruments)
    by_type = {}
    failures = []
    for ins in instruments:
        t = (ins.get("type") or "unknown")
        t = t.strip() if isinstance(t, str) else "unknown"
        by_type[t] = by_type.get(t, 0) + 1

        issues = []
        details = {}
        pf = ins.get("pass_fail") or {}
        meas = ins.get("measured") or {}
        orient = ins.get("orientation") or {}
        if pf.get("upstream") is False:
            issues.append("upstream_fail")
            details["upstream_m"] = meas.get("upstream_m")
        if pf.get("downstream") is False:
            issues.append("downstream_fail")
            details["downstream_m"] = meas.get("downstream_m")
        if orient and (orient.get("vertical_pass") is False or (isinstance(orient.get("tilt_deg"), (int,float)) and abs(orient.get("tilt_deg")) > 3.0)):
            issues.append("orientation_fail")
            details["tilt_deg"] = orient.get("tilt_deg")
        if meas.get("upstream_m") is None or meas.get("downstream_m") is None:
            issues.append("missing_geometry")
        if issues:
            failures.append({
                "tag": ins.get("tag") or ins.get("Tag") or "UNKNOWN",
                "type": t,
                "issues": issues,
                "details": details
            })

    by_type_list = [{"type": k, "count": v} for k, v in sorted(by_type.items(), key=lambda x: -x[1])]
    recommendations = []
    if failures:
        recommendations.append("Fix instruments with upstream/downstream fails by adding straight spool lengths per spec.")
    else:
        recommendations.append("No major pass/fail issues detected.")
    return {
        "total_instruments": total,
        "by_type": by_type_list,
        "failures": failures,
        "summary_recommendations": recommendations,
        "confidence": "low"
    }

def ensure_genai_client():
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai client is not installed in the environment.")
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None
    try:
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
            except TypeError:
                client = genai.Client(key=api_key)
        else:
            client = genai.Client()
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to construct genai client: {e}")

def _safe_close_client(client):
    """
    Attempt to close the genai client without letting unhandled async exceptions bubble up.
    - Prefer client.close() if available.
    - If only client.aclose() exists, schedule it safely with a done-callback to catch exceptions.
    - If we are in a non-running loop, run it synchronously.
    """
    try:
        if client is None:
            return
        if hasattr(client, "close") and callable(getattr(client, "close")):
            try:
                client.close()
            except Exception as e:
                logger.warning("genai client.close() raised: %s", e)
            return

        if hasattr(client, "aclose") and callable(getattr(client, "aclose")):
            aclose_func = client.aclose
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                try:
                    task = asyncio.create_task(aclose_func())
                    def _done_cb(fut):
                        try:
                            exc = fut.exception()
                            if exc:
                                logger.warning("genai client.aclose() raised: %s", exc)
                        except asyncio.CancelledError:
                            pass
                    task.add_done_callback(_done_cb)
                except Exception as e:
                    logger.warning("Failed to schedule genai.aclose(): %s", e)
            else:
                try:
                    if loop:
                        loop.run_until_complete(aclose_func())
                    else:
                        new_loop = asyncio.new_event_loop()
                        try:
                            new_loop.run_until_complete(aclose_func())
                        finally:
                            new_loop.close()
                except Exception as e:
                    logger.warning("genai client.aclose() (sync) raised: %s", e)
            return
    except Exception as outer:
        logger.warning("Exception while trying to close genai client: %s", outer)

@router.post("/summarize-instruments")
async def summarize_instruments(payload: Dict[str, Any]):
    try:
        instruments = payload.get("instruments")
        if not isinstance(instruments, list):
            raise HTTPException(status_code=400, detail="`instruments` must be a list")

        brief = [_make_brief(ins) for ins in instruments]

        if not GENAI_AVAILABLE:
            return {"ok": True, "ai_result": _simple_local_summary(brief), "raw": None, "note": "GENAI client not installed - returned local summary"}

        try:
            client = ensure_genai_client()
        except Exception as e:
            logger.warning("Could not construct genai client: %s", e)
            return {"ok": True, "ai_result": _simple_local_summary(brief), "raw": None, "note": f"Failed to create genai client: {e}"}

        resp = None
        try:
            body = PROMPT_TEMPLATE + "\n\n" + json.dumps({"instruments": brief}, indent=2)
            resp = client.models.generate_content(
                model="gemini-2.5",
                contents=body,
                max_output_tokens=800
            )
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
            parsed = None
            s = text.strip()
            if s.startswith("{") or s.startswith("["):
                parsed = json.loads(s)
            else:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(s[start:end+1])
            if parsed is None:
                return {"ok": False, "error": "AI returned non-JSON", "raw": text}
            return {"ok": True, "ai_result": parsed, "raw": text}
        except Exception as e:
            logger.exception("genai model call failed")
            return {"ok": False, "error": str(e), "raw": None}
        finally:
            try:
                _safe_close_client(client)
            except Exception as e:
                logger.warning("Error during genai client cleanup: %s", e)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
