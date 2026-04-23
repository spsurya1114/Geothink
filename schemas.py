# schemas.py
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional, Any
from enum import Enum


# --- Allowed GIS operations ---
# This is a fixed list. The LLM can ONLY use these operation names.
# If it tries to invent something like "arcpy_analysis", 
# Pydantic will reject it immediately.
class GISOperation(str, Enum):
    FETCH_DEM          = "fetch_dem"
    REPROJECT          = "reproject"
    CLIP_TO_BOUNDARY   = "clip_to_boundary"
    FILL_DEPRESSIONS   = "fill_depressions"
    FLOW_DIRECTION     = "flow_direction"
    FLOW_ACCUMULATION  = "flow_accumulation"
    EXTRACT_STREAMS    = "extract_streams"
    HAND_ANALYSIS      = "hand_analysis"
    THRESHOLD_CLASSIFY = "threshold_classify"
    VECTOR_OVERLAY     = "vector_overlay"
    EXPORT_RESULT      = "export_result"


# --- Allowed coordinate systems ---
# India uses EPSG:32643 and EPSG:32644 (UTM zones)
# Trichy specifically falls in EPSG:32644
VALID_CRS = {
    "EPSG:4326",   # standard lat/lon (WGS84)
    "EPSG:32643",  # UTM Zone 43N (western India)
    "EPSG:32644",  # UTM Zone 44N (eastern India — Trichy is here)
    "EPSG:3857",   # web mercator (for map display)
}


# --- A single step in the workflow ---
class GISStep(BaseModel):
    step_id:     int
    operation:   GISOperation       # must be from the allowed list above
    description: str
    inputs:      dict[str, Any]     # what this step needs as input
    outputs:     dict[str, str]     # what this step produces
    depends_on:  list[int] = []     # which step_ids must finish first
    crs:         Optional[str] = None

    @field_validator("crs")
    @classmethod
    def crs_must_be_valid(cls, v):
        if v and v not in VALID_CRS:
            raise ValueError(
                f"CRS '{v}' is not allowed. "
                f"Must be one of: {VALID_CRS}"
            )
        return v


# --- The full workflow (a list of steps) ---
class GISWorkflow(BaseModel):
    query:              str
    region:             str
    steps:              list[GISStep]
    expected_output:    Literal["flood_risk_map", "risk_report", "both"]
    reasoning_summary:  str   # the LLM explains its thinking here

    @model_validator(mode="after")
    def validate_step_order_and_dependencies(self):
        ids = [s.step_id for s in self.steps]

        # Step IDs must be in order: 1, 2, 3 ...
        if ids != sorted(ids):
            raise ValueError("step_id values must be in ascending order")

        # Every depends_on reference must point to a real step
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step {step.step_id} depends on step {dep}, "
                        f"which does not exist"
                    )
        return self