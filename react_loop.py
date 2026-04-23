# react_loop.py
import json
from schemas import GISWorkflow
from llm import plan_workflow
from pydantic import ValidationError

MAX_RETRIES = 3

REPAIR_PROMPT = """
The previous GIS workflow JSON failed with this error:

ERROR:
{error}

FAILED JSON:
{failed_json}

You must produce a corrected workflow JSON that fixes this exact error.

Common fixes:
- If error mentions CRS mismatch -> add a reproject step before the failing step
- If error mentions unknown operation -> replace with one of the allowed operations
- If error mentions missing depends_on -> add the correct step_id reference
- If error mentions step_id not sequential -> renumber steps starting from 1
- If error mentions FileNotFoundError -> check that previous steps produce the right output keys

Allowed operations: fetch_dem, reproject, clip_to_boundary, fill_depressions,
flow_direction, flow_accumulation, extract_streams, hand_analysis,
threshold_classify, vector_overlay, export_result

Allowed CRS: EPSG:4326, EPSG:32644, EPSG:32643, EPSG:3857

Respond with corrected raw JSON only. No explanation, no markdown fences.
"""


async def react_loop(
    query:       str,
    region:      str,
    failed_json: dict,
    error_msg:   str,
    doc_context: str,
) -> GISWorkflow:
    """
    Reason-Act-Observe loop.
    Tries to self-heal a broken workflow up to MAX_RETRIES times.

    Each iteration:
      REASON  -> understand what went wrong
      ACT     -> ask LLM to fix it
      OBSERVE -> validate the fix, loop if still broken
    """
    error_history = []

    for attempt in range(1, MAX_RETRIES + 1):

        print(f"\n[ReAct] {'='*40}")
        print(f"[ReAct] Self-heal attempt {attempt}/{MAX_RETRIES}")
        print(f"[ReAct] Error: {error_msg}")

        # ── REASON ──────────────────────────────────
        # Diagnose what kind of error this is
        error_type = _diagnose_error(error_msg)
        print(f"[ReAct] Diagnosed as: {error_type}")

        # ── ACT ─────────────────────────────────────
        # Build a repair prompt and ask the LLM to fix it
        repair_prompt = REPAIR_PROMPT.format(
            error=error_msg,
            failed_json=json.dumps(failed_json, indent=2),
        )

        try:
            repaired_json = await plan_workflow(
                repair_prompt,
                region,
                doc_context
            )
        except RuntimeError as e:
            print(f"[ReAct] LLM call failed during repair: {e}")
            error_history.append(f"Attempt {attempt}: LLM failed — {e}")
            continue

        # ── OBSERVE ──────────────────────────────────
        # Check if the repair actually fixed the problem
        try:
            workflow = GISWorkflow.model_validate(repaired_json)

            # Success — annotate the reasoning summary with repair info
            workflow.reasoning_summary += (
                f" [Auto-repaired after {attempt} attempt(s). "
                f"Original error: {error_msg[:100]}]"
            )

            print(f"[ReAct] Self-healed successfully on attempt {attempt}")
            print(f"[ReAct] Fixed workflow has {len(workflow.steps)} steps")
            return workflow

        except ValidationError as new_error:
            new_error_msg = str(new_error)
            print(f"[ReAct] Repair attempt {attempt} still invalid: "
                  f"{new_error_msg[:150]}")
            error_history.append(
                f"Attempt {attempt}: {new_error_msg[:150]}"
            )
            # Feed the new error into the next iteration
            error_msg  = new_error_msg
            failed_json = repaired_json

        except Exception as e:
            print(f"[ReAct] Unexpected error on attempt {attempt}: {e}")
            error_history.append(f"Attempt {attempt}: {e}")
            error_msg = str(e)

    # All retries exhausted
    raise RuntimeError(
        f"GeoThink could not self-heal after {MAX_RETRIES} attempts.\n"
        f"Error chain:\n" +
        "\n".join(f"  {h}" for h in error_history)
    )


def _diagnose_error(error_msg: str) -> str:
    """
    Categorize the error so we can log what kind of
    problem the ReAct loop is fixing.
    """
    error_lower = error_msg.lower()

    if "crs" in error_lower or "projection" in error_lower:
        return "CRS_MISMATCH"
    elif "operation" in error_lower or "not a valid" in error_lower:
        return "INVALID_OPERATION"
    elif "step_id" in error_lower or "sequential" in error_lower:
        return "STEP_ORDER_ERROR"
    elif "depends_on" in error_lower:
        return "DEPENDENCY_ERROR"
    elif "filenotfound" in error_lower or "no such file" in error_lower:
        return "MISSING_FILE"
    elif "json" in error_lower:
        return "JSON_PARSE_ERROR"
    else:
        return "UNKNOWN_ERROR"