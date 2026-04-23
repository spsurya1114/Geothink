# llm.py
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()   # reads your .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
You are a GIS workflow planner for flood risk modeling in India.

Your job is to convert a natural language query into a structured 
JSON workflow. You must respond with ONLY raw JSON — no explanation, 
no markdown code fences, no extra text before or after.

The JSON must follow this exact structure:
{
  "query": "<the original user query>",
  "region": "<region name>",
  "reasoning_summary": "<1-2 sentences explaining your plan>",
  "expected_output": "flood_risk_map",
  "steps": [
    {
      "step_id": 1,
      "operation": "<operation name>",
      "description": "<what this step does>",
      "inputs": {},
      "outputs": {},
      "depends_on": [],
      "crs": "EPSG:4326"
    }
  ]
}

ALLOWED OPERATIONS (use ONLY these exact strings):
- fetch_dem
- reproject
- clip_to_boundary
- fill_depressions
- flow_direction
- flow_accumulation
- extract_streams
- hand_analysis
- threshold_classify
- vector_overlay
- export_result

ALLOWED CRS VALUES (use ONLY these):
- EPSG:4326  (use this for fetch_dem step)
- EPSG:32644 (use this for all processing steps — correct for Trichy/Tamil Nadu)
- EPSG:32643 (UTM zone 43N — western India)
- EPSG:3857  (use only for export/display steps)

RULES:
1. Always start with fetch_dem as step 1
2. Always reproject to EPSG:32644 as step 2 (unless EPSG:4326 is requested)
3. step_id must be sequential: 1, 2, 3 ...
4. depends_on must reference valid step_ids that come before
5. IMPORTANT: 'threshold_classify' now uses HAND (Height Above Nearest Drainage). 
   You MUST choose `low_m` and `high_m` dynamically based on the geographic region being queried! 
   - Flat, coastal, or delta areas (e.g., Chennai, Thanjavur): use strict thresholds (e.g., `low_m: 2`, `high_m: 5`).
   - Hilly or mountainous areas (e.g., Ooty, Kodaikanal): use higher thresholds (e.g., `low_m: 10`, `high_m: 20`).
   - Standard plains or plateaus (e.g., Madurai, Trichy): use moderate thresholds (e.g., `low_m: 5`, `high_m: 15`).
   Place `low_m` and `high_m` inside the `inputs` dictionary for `threshold_classify`.
6. For 'fetch_dem' and 'vector_overlay', you MUST pass the requested Region name in 'inputs' as {"place_name": "<Region>, Tamil Nadu, India"}.
7. Respond with raw JSON only — nothing else
"""


async def plan_workflow(query: str, region: str, doc_context: str = "") -> dict:
    """
    Send the user's query to Groq (Llama 3 70B) and get back a workflow JSON.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Make sure you created a .env file with your key."
        )

    if doc_context:
        user_message = f"""
Use this GIS documentation as reference for correct parameters:

{doc_context}

---

User query: {query}
Region: {region}

Generate the workflow JSON now.
"""
    else:
        user_message = f"""
User query: {query}
Region: {region}

Generate the workflow JSON now.
"""

    print(f"\n[LLM] Sending query to Groq: '{query}'")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",   # free on Groq
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.1,   # low = more consistent JSON output
                }
            )
            response.raise_for_status()

        raw_text = response.json()["choices"][0]["message"]["content"]
        print(f"[LLM] Raw response received ({len(raw_text)} chars)")

        # Clean up any accidental markdown fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        workflow_dict = json.loads(cleaned)
        print(f"[LLM] Successfully parsed JSON with "
              f"{len(workflow_dict.get('steps', []))} steps")
        return workflow_dict

    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Groq API returned an error: {e.response.status_code} "
            f"{e.response.text}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM did not return valid JSON.\n"
            f"Error: {e}\n"
            f"Raw response:\n{raw_text}"
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")