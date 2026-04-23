# agents.py
import json
import httpx
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from rag import enrich_with_docs

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class PipelineState:
    query:            str = ""
    region:           str = ""
    doc_context:      str = ""
    relevant_tools:   list = field(default_factory=list)
    raw_workflow:     Optional[dict] = None
    reasoning:        str = ""
    validated:        bool = False
    validation_error: Optional[str] = None
    execution_error:  Optional[str] = None
    execution_result: Optional[dict] = None
    cot_log:          list = field(default_factory=list)
    repair_attempts:  int = 0
    repair_history:   list = field(default_factory=list)


class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name          = name
        self.system_prompt = system_prompt

    async def call_llm(self, user_message: str) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.1,
                }
            )
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError


class LibrarianAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Librarian",
            system_prompt="""
You are a GIS documentation specialist.
Given a flood risk query, identify which tools are needed.
Return ONLY this JSON:
{
  "relevant_tools": ["whitebox", "rasterio", "geopandas"],
  "key_parameters": {
    "flow_accumulation_threshold": 50,
    "elevation_low_risk_m": 75,
    "elevation_high_risk_m": 100,
    "target_crs": "EPSG:32644"
  },
  "reasoning": "one sentence"
}
Raw JSON only, no markdown.
"""
        )

    async def run(self, state: PipelineState) -> PipelineState:
        print(f"\n[Librarian] Analyzing query...")

        response = await self.call_llm(
            f"Query: {state.query}\nRegion: {state.region}\n"
            f"Which GIS tools are needed?"
        )

        try:
            cleaned  = response.strip().strip("```json").strip("```").strip()
            tool_info = json.loads(cleaned)
            state.relevant_tools = tool_info.get("relevant_tools", ["whitebox"])
            key_params           = tool_info.get("key_parameters", {})
            print(f"[Librarian] Tools: {state.relevant_tools}")
        except Exception:
            state.relevant_tools = ["whitebox"]
            key_params           = {}
            print(f"[Librarian] Using default tools")

        # Fetch real documentation via Firecrawl
        state.doc_context = await enrich_with_docs(state.query)

        if key_params:
            state.doc_context += (
                f"\n\n## Recommended parameters\n"
                f"{json.dumps(key_params, indent=2)}"
            )

        print(f"[Librarian] Documentation ready ({len(state.doc_context)} chars)")
        return state


class DirectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Director",
            system_prompt="""
You are a GIS workflow director for flood risk modeling in India.
Convert natural language queries into structured JSON workflows.

JSON structure:
{
  "query": "<original query>",
  "region": "<region>",
  "reasoning_summary": "<1-2 sentence plan>",
  "expected_output": "flood_risk_map",
  "steps": [
    {
      "step_id": 1,
      "operation": "<operation>",
      "description": "<what this step does>",
      "inputs": {},
      "outputs": {},
      "depends_on": [],
      "crs": "EPSG:4326"
    }
  ]
}

ALLOWED OPERATIONS ONLY:
fetch_dem, reproject, clip_to_boundary, fill_depressions,
flow_direction, flow_accumulation, extract_streams,
hand_analysis, threshold_classify, vector_overlay, export_result

ALLOWED CRS: EPSG:4326, EPSG:32644, EPSG:32643, EPSG:3857

RULES:
1. Always start with fetch_dem as step 1
2. Always reproject to EPSG:32644 as step 2
3. step_id must be sequential starting from 1
4. depends_on must reference valid prior step_ids
5. Raw JSON only
"""
        )

    async def run(self, state: PipelineState) -> PipelineState:
        print(f"\n[Director] Planning workflow...")

        prompt = f"""
Documentation and parameters:
{state.doc_context}

---
Query: {state.query}
Region: {state.region}

Generate workflow JSON now.
"""
        response = await self.call_llm(prompt)

        cleaned = response.strip()
        for fence in ["```json", "```"]:
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            state.raw_workflow = json.loads(cleaned)
            steps              = len(state.raw_workflow.get("steps", []))
            state.reasoning    = state.raw_workflow.get("reasoning_summary", "")
            print(f"[Director] Planned {steps} steps")
            print(f"[Director] Reasoning: {state.reasoning}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Director produced invalid JSON: {e}")

        return state


class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Validator",
            system_prompt="""
You are a GIS workflow repair specialist.
Fix broken workflow JSON based on the error message.
Return corrected raw JSON only.
"""
        )

    async def run(self, state: PipelineState) -> PipelineState:
        from schemas import GISWorkflow
        from pydantic import ValidationError

        print(f"\n[Validator] Validating workflow...")

        MAX_RETRIES = 3

        for attempt in range(1, MAX_RETRIES + 1):
            if state.execution_error:
                error_msg = f"Runtime Execution Error: {state.execution_error}"
                state.execution_error = None # Clear so we don't skip validation next loop
            else:
                try:
                    GISWorkflow.model_validate(state.raw_workflow)
                    state.validated = True
                    print(f"[Validator] Passed on attempt {attempt}")
                    return state

                except ValidationError as e:
                    error_msg = str(e)

            print(f"[Validator] Attempt {attempt} failed: {error_msg[:100]}")

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Validation failed after {MAX_RETRIES} attempts: {error_msg}"
                )

            # ReAct repair
            state.repair_attempts += 1
            repair_prompt = f"""
Fix this workflow JSON:
ERROR: {error_msg}
BROKEN JSON:
{json.dumps(state.raw_workflow, indent=2)}
Return corrected JSON only.
"""
            response = await self.call_llm(repair_prompt)
            cleaned  = response.strip().strip("```json").strip("```").strip()

            try:
                state.raw_workflow = json.loads(cleaned)
                state.repair_history.append(f"Attempt {attempt}: {error_msg[:80]}")
                print(f"[Validator] Repair applied, retrying...")
            except json.JSONDecodeError:
                print(f"[Validator] Repair was not valid JSON, retrying...")

        return state


class GeoThinkOrchestrator:
    """
    Runs the full Director-Librarian-Validator-Executor pipeline.
    Accepts an optional log_callback for streaming logs to the UI.
    """
    def __init__(self):
        self.librarian = LibrarianAgent()
        self.director  = DirectorAgent()
        self.validator = ValidatorAgent()

    async def run(
        self,
        query:        str,
        region:       str,
        log_callback  = None   # called with each log line for live UI updates
    ) -> dict:

        def log(msg: str):
            print(msg)
            if log_callback:
                log_callback(msg)

        log(f"\n{'='*50}")
        log(f"[Orchestrator] Query:  '{query}'")
        log(f"[Orchestrator] Region: '{region}'")
        log(f"{'='*50}")

        state = PipelineState(query=query, region=region)

        state = await self.librarian.run(state)
        log(f"[Librarian] ✓ Ready — {len(state.doc_context)} chars of context")
        log(f"[Librarian] Tools identified: {state.relevant_tools}")

        state = await self.director.run(state)
        steps = len(state.raw_workflow.get('steps', []))
        log(f"[Director] ✓ Planned {steps}-step workflow")
        log(f"[Director] Reasoning: {state.reasoning}")

        MAX_EXEC_RETRIES = 3
        
        for exec_attempt in range(1, MAX_EXEC_RETRIES + 1):
            state = await self.validator.run(state)
            log(f"[Validator] ✓ Schema validated")
            if state.repair_attempts > 0:
                log(f"[Validator] Self-healed {state.repair_attempts} time(s)")

            log(f"\n[Executor] Starting GIS pipeline (Attempt {exec_attempt})...")
            from schemas import GISWorkflow
            from executor import execute_workflow

            workflow = GISWorkflow.model_validate(state.raw_workflow)

            # Patch executor to stream logs too
            import executor as exc
            original_dispatch = exc._dispatch

            def logging_dispatch(step, context):
                log(f"[Executor] Step {step.step_id}: {step.operation.value}")
                result = original_dispatch(step, context)
                log(f"[Executor] Step {step.step_id} ✓")
                return result

            exc._dispatch = logging_dispatch
            
            try:
                result = await execute_workflow(workflow)
                exc._dispatch = original_dispatch  # restore
                log(f"\n[Executor] ✓ All steps complete")
                break # Success!
            except RuntimeError as e:
                exc._dispatch = original_dispatch
                log(f"[Executor] ERROR: {e}")
                
                if exec_attempt == MAX_EXEC_RETRIES:
                    raise RuntimeError(f"Pipeline failed after {MAX_EXEC_RETRIES} execution attempts: {e}")
                
                log(f"[Orchestrator] Execution failed. Passing back to Validator for autonomous repair...")
                state.execution_error = str(e)
                state.validated = False

        log(f"[Orchestrator] ✓ Pipeline finished successfully")

        # Add multi-agent metadata
        result["agents_used"]     = ["Librarian", "Director", "Validator", "Executor"]
        result["repair_attempts"] = state.repair_attempts
        result["reasoning"]       = state.reasoning
        result["tools_used"]      = state.relevant_tools

        return result