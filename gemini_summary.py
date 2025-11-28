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

When asked a question, produce a helpful, concise answer that references instruments by tag where appropriate,
and include clear improvement actions when possible.

When asked for a summary you MUST return a JSON object with fields:
  - total_instruments (int)
  - by_type (array of {type: string, count: int})
  - failures (array of {tag, type, issues, details})
  - summary_recommendations (array of strings)
Return only JSON (no extra explanation).
"""

CHAT_PROMPT_TEMPLATE = """
You are a senior piping instrumentation QA engineer.
Context: a JSON array "instruments" is provided.

When answering a user's question you MUST return a JSON object with fields:
  - answer (string)
  - recommendations (array of strings)
  - references (optional array of instrument tags referenced)
  - confidence (string; e.g. "low","medium","high")

Return only JSON (no extra explanation).
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
    api_key = "AIzaSyDsqJRbVAnefvQEge_GWgrqOHqD6XcB7y8"
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

def genai_generate(client, model, contents, max_tokens=800):
    try:
        if hasattr(client, "responses") and hasattr(client.responses, "generate"):
            try:
                return client.responses.generate(model=model, input=contents, max_output_tokens=max_tokens)
            except TypeError:
                return client.responses.generate(model=model, input=contents, max_tokens=max_tokens)
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            try:
                return client.responses.create(model=model, input=contents, max_output_tokens=max_tokens)
            except TypeError:
                return client.responses.create(model=model, input=contents, max_tokens=max_tokens)
        if hasattr(client, "models") and hasattr(client.models, "generate"):
            try:
                return client.models.generate(model=model, content=contents, max_output_tokens=max_tokens)
            except TypeError:
                return client.models.generate(model=model, content=contents, max_tokens=max_tokens)
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            try:
                return client.models.generate_content(model=model, contents=contents, max_output_tokens=max_tokens)
            except TypeError:
                try:
                    return client.models.generate_content(model=model, contents=contents, max_tokens=max_tokens)
                except TypeError:
                    return client.models.generate_content(model=model, contents=contents)
    except Exception as e:
        raise e
    raise RuntimeError("No supported genai model invocation method found on client.")

def extract_text_from_response(resp):
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
        output = getattr(resp, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(output, list) and len(output) > 0:
            parts = []
            for item in output:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if isinstance(c, dict) and "text" in c:
                                parts.append(c["text"])
                            elif isinstance(c, str):
                                parts.append(c)
                    elif "text" in item:
                        parts.append(item["text"])
            if parts:
                return "\n".join(parts)
        if isinstance(resp, dict):
            if "output" in resp:
                out = resp["output"]
                if isinstance(out, str):
                    return out
                if isinstance(out, list):
                    collected = []
                    for it in out:
                        if isinstance(it, str):
                            collected.append(it)
                        elif isinstance(it, dict):
                            if "content" in it and isinstance(it["content"], list):
                                for c in it["content"]:
                                    if isinstance(c, dict) and "text" in c:
                                        collected.append(c["text"])
                                    elif isinstance(c, str):
                                        collected.append(c)
                            elif "text" in it:
                                collected.append(it["text"])
                    if collected:
                        return "\n".join(collected)
            if "text" in resp:
                return resp["text"]
            if "answer" in resp:
                return resp["answer"]
        return str(resp)
    except Exception as e:
        logger.warning("Failed to extract text from response: %s", e)
        return str(resp)

async def _ask_model_for_json(client, model, initial_text, schema_type="summary"):

    if schema_type == "summary":
        enforce_instructions = (
            "Convert the assistant response below into a JSON object with fields:\n"
            "  - total_instruments (int)\n"
            "  - by_type (array of {type: string, count: int})\n"
            "  - failures (array of {tag, type, issues, details})\n"
            "  - summary_recommendations (array of strings)\n"
            "Return only valid JSON and nothing else."
        )
    else:
        enforce_instructions = (
            "Convert the assistant response below into a JSON object with fields:\n"
            "  - answer (string)\n"
            "  - recommendations (array of strings)\n"
            "  - references (array of tags) (optional)\n"
            "  - confidence (string)\n"
            "Return only valid JSON and nothing else."
        )

    prompt = f"{enforce_instructions}\n\nAssistant response:\n{initial_text}\n\nNow output the JSON."
    try:
        resp2 = genai_generate(client, model=model, contents=prompt, max_tokens=400)
        text2 = extract_text_from_response(resp2)
        return text2
    except Exception as e:
        logger.warning("Second-pass JSON enforcement call failed: %s", e)
        return None

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

        try:
            body = PROMPT_TEMPLATE + "\n\n" + json.dumps({"instruments": brief}, indent=2)
            resp = genai_generate(client, model="gemini-2.5-flash", contents=body, max_tokens=800)
            text = extract_text_from_response(resp)
            parsed = None
            s = (text or "").strip()
            try:
                if s.startswith("{") or s.startswith("["):
                    parsed = json.loads(s)
                else:
                    start = s.find("{")
                    end = s.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(s[start:end+1])
            except Exception:
                parsed = None

            if parsed is None:
                logger.info("Primary parse failed for summarize-instruments — asking model to output strict JSON.")
                text2 = await _ask_model_for_json(client, model="gemini-2.5-flash", initial_text=text or "", schema_type="summary")
                s2 = (text2 or "").strip()
                try:
                    if s2.startswith("{") or s2.startswith("["):
                        parsed = json.loads(s2)
                    else:
                        start = s2.find("{")
                        end = s2.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            parsed = json.loads(s2[start:end+1])
                except Exception:
                    parsed = None
                if parsed is not None:
                    return {"ok": True, "ai_result": parsed, "raw": text2}
                logger.warning("Model failed to return JSON even after enforcement; returning local summary fallback.")
                return {"ok": True, "ai_result": _simple_local_summary(brief), "raw": text, "note": "fallback to local summary; model did not produce JSON"}

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


@router.post("/chat")
async def ai_chat(payload: Dict[str, Any]):
    try:
        question = (payload or {}).get("question")
        if not question or not isinstance(question, str):
            raise HTTPException(status_code=400, detail="`question` (string) is required")

        instruments = payload.get("instruments")
        if instruments is None:
            report = payload.get("report") or {}
            instruments = report.get("instruments") if isinstance(report, dict) else None
        if not isinstance(instruments, list):
            instruments = []

        brief = [_make_brief(ins) for ins in instruments]
        history = payload.get("history") or []

        if not GENAI_AVAILABLE:
            summary = _simple_local_summary(brief)
            answer_lines = []
            answer_lines.append(f"I analyzed {summary['total_instruments']} instrument(s).")
            if summary["failures"]:
                answer_lines.append(f"{len(summary['failures'])} instruments have issues: " +
                                    ", ".join([f['tag'] for f in summary['failures'][:8]]))
                answer_lines.append("Top recommendations:")
                answer_lines += summary["summary_recommendations"]
            else:
                answer_lines.append("No major pass/fail issues detected.")
                answer_lines += summary["summary_recommendations"]
            q = question.lower()
            if "why" in q or "reason" in q:
                answer_lines.append("Common reasons: insufficient straight-run upstream/downstream, incorrect orientation, missing geometry.")
            if "how" in q or "fix" in q or "recommend" in q:
                answer_lines.append("General fix actions: add needed upstream/downstream straight spool lengths, verify pipe diameter and alignment, check placement coordinates in the model.")
            for f in summary["failures"]:
                if f["tag"] and f["tag"].lower() in q:
                    answer_lines.append(f"Details for {f['tag']}: issues={','.join(f['issues'])}, {json.dumps(f.get('details',{}))}")
            return {
                "ok": True,
                "answer": "\n".join(answer_lines),
                "recommendations": summary.get("summary_recommendations", []),
                "raw": None,
                "confidence": summary.get("confidence", "low")
            }

        try:
            client = ensure_genai_client()
        except Exception as e:
            logger.warning("Could not construct genai client for chat: %s", e)
            return {"ok": False, "error": f"Failed to create genai client: {e}"}

        try:
            system = CHAT_PROMPT_TEMPLATE
            context_part = json.dumps({"instruments": brief}, indent=2)[:8000]
            full_prompt = f"{system}\n\nContext (instruments):\n{context_part}\n\nUser question:\n{question}\n\nReturn only JSON."
            resp = genai_generate(client, model="gemini-2.5-flash", contents=full_prompt, max_tokens=800)
            text = extract_text_from_response(resp)
            parsed = None
            s = (text or "").strip()
            try:
                if s.startswith("{") or s.startswith("["):
                    parsed = json.loads(s)
                else:
                    start = s.find("{")
                    end = s.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(s[start:end+1])
            except Exception:
                parsed = None

            if parsed is None:
                logger.info("Primary parse failed for /ai/chat — asking model to output strict JSON.")
                text2 = await _ask_model_for_json(client, model="gemini-2.5-flash", initial_text=text or "", schema_type="chat")
                s2 = (text2 or "").strip()
                try:
                    if s2.startswith("{") or s2.startswith("["):
                        parsed = json.loads(s2)
                    else:
                        start = s2.find("{")
                        end = s2.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            parsed = json.loads(s2[start:end+1])
                except Exception:
                    parsed = None
                if parsed is None:
                    logger.warning("Model failed to return JSON for chat even after enforcement; returning plain text as answer (but still JSON-wrapped).")
                    return {"ok": True, "answer": text or "", "recommendations": [], "raw": text, "confidence": "low"}

            answer = parsed.get("answer") or parsed.get("text") or parsed.get("response") or str(parsed)
            recs = parsed.get("recommendations") or parsed.get("summary_recommendations") or []
            refs = parsed.get("references") or parsed.get("tags") or []
            conf = parsed.get("confidence") or "medium"
            return {"ok": True, "answer": answer, "recommendations": recs, "references": refs, "raw": text, "confidence": conf}
        except Exception as e:
            logger.exception("genai model call failed in /ai/chat")
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
