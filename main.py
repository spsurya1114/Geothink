# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schemas import GISWorkflow
from llm import plan_workflow
from rag import enrich_with_docs
from executor import execute_workflow
from react_loop import react_loop         
from pydantic import ValidationError

app = FastAPI(title="GeoThink API")


@app.get("/health")
def health():
    return {"status": "ok", "message": "GeoThink backend is running"}


@app.post("/execute")
async def execute(payload: dict):
    query  = payload.get("query", "")
    region = payload.get("region", "Trichy")

    if not query:
        raise HTTPException(status_code=400, detail="query field is required")

    print(f"\n{'='*50}")
    print(f"New request: '{query}' for region: '{region}'")
    print(f"{'='*50}")

    # Step 1: RAG
    doc_context = await enrich_with_docs(query)

    # Step 2: LLM planning
    try:
        raw_workflow = await plan_workflow(query, region, doc_context)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 3: Validate — if it fails, try to self-heal via ReAct
    try:
        workflow = GISWorkflow.model_validate(raw_workflow)
        print(f"[Validator] Workflow passed — {len(workflow.steps)} steps")

    except ValidationError as e:
        print(f"[Validator] Failed — attempting self-heal via ReAct loop")

        try:
            # Hand off to the ReAct loop for autonomous repair
            workflow = await react_loop(
                query=query,
                region=region,
                failed_json=raw_workflow,
                error_msg=str(e),
                doc_context=doc_context,
            )
            print(f"[ReAct] Recovered successfully")

        except RuntimeError as react_error:
            # ReAct exhausted all retries — give up gracefully
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Could not generate a valid workflow",
                    "error":   str(react_error),
                }
            )

    # Step 4: Execute
    try:
        result = await execute_workflow(workflow)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


@app.post("/validate")
async def validate_only(payload: dict):
    try:
        workflow = GISWorkflow.model_validate(payload)
        return {"valid": True, "steps": len(workflow.steps)}
    except ValidationError as e:
        return {"valid": False, "error": str(e)}