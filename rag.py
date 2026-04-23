# rag.py
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_URL     = "https://api.firecrawl.dev/v1/scrape"

# These are the actual documentation pages we fetch from.
# Each key maps to a real URL with useful parameter information.
DOC_SOURCES = {
    "whitebox": "https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html",
    "geopandas": "https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html",
    "rasterio":  "https://rasterio.readthedocs.io/en/latest/api/rasterio.html",
}

# Keywords that tell us which doc to fetch for a given query
KEYWORD_MAP = {
    "whitebox": [
        "dem", "elevation", "hand", "flow", "drainage",
        "watershed", "breach", "depression", "stream", "terrain"
    ],
    "geopandas": [
        "vector", "shapefile", "join", "clip",
        "dissolve", "geodataframe", "boundary"
    ],
    "rasterio": [
        "raster", "tif", "pixel", "band",
        "reproject", "mask", "classify"
    ],
}


async def fetch_doc(url: str) -> str:
    """
    Use Firecrawl to fetch a documentation page
    and return clean markdown text.
    """
    if not FIRECRAWL_API_KEY:
        print("[RAG] WARNING: No Firecrawl key found, skipping doc fetch")
        return ""

    print(f"[RAG] Fetching: {url}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                FIRECRAWL_URL,
                headers={
                    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "url":             url,
                    "formats":         ["markdown"],
                    "onlyMainContent": True,   # strips nav, footer, ads
                }
            )
            response.raise_for_status()

        data     = response.json()
        markdown = data.get("data", {}).get("markdown", "")
        print(f"[RAG] Fetched {len(markdown)} chars from {url}")
        return markdown

    except Exception as e:
        # If Firecrawl fails for any reason, we just
        # continue without the doc context — graceful degradation
        print(f"[RAG] Failed to fetch {url}: {e}")
        return ""


async def enrich_with_docs(query: str) -> str:
    """
    1. Look at the query keywords
    2. Decide which docs are relevant
    3. Fetch them via Firecrawl
    4. Return a combined context string for the LLM prompt
    """
    query_lower = query.lower()

    # Figure out which doc sources are relevant to this query
    relevant_sources = []
    for source_name, keywords in KEYWORD_MAP.items():
        if any(keyword in query_lower for keyword in keywords):
            relevant_sources.append(source_name)

    # Default to whitebox if nothing matched
    # (most flood queries will need it anyway)
    if not relevant_sources:
        relevant_sources = ["whitebox"]

    print(f"[RAG] Relevant sources for query: {relevant_sources}")

    # Fetch the docs (cap at 2 to stay within token budget)
    context_parts = []
    for source in relevant_sources[:2]:
        url  = DOC_SOURCES[source]
        text = await fetch_doc(url)

        if text:
            # Trim to 2000 chars — enough for parameter hints
            # without overwhelming the LLM context window
            trimmed = text[:2000]
            context_parts.append(
                f"## {source} documentation\n{trimmed}"
            )

    if not context_parts:
        print("[RAG] No context retrieved — LLM will plan without docs")
        return ""

    # Append GeoThink specific tool guidelines
    geothink_docs = """
## GeoThink Specific Guidelines
For the `threshold_classify` operation, the system computes HAND (Height Above Nearest Drainage).
- A cell with HAND = 0 is a river/stream.
- A cell with HAND = 3 means it is 3 meters vertically higher than the nearest stream.
- The `threshold_classify` step expects `low_m` and `high_m` as inputs.
- IMPORTANT: You MUST choose `low_m` and `high_m` dynamically based on the geographic region being queried! 
  - For flat, coastal, or delta areas (e.g., Chennai, Thanjavur), use strict thresholds (e.g., `low_m: 2`, `high_m: 5`).
  - For hilly or mountainous areas (e.g., Ooty, Kodaikanal), use higher thresholds (e.g., `low_m: 10`, `high_m: 20`).
  - For standard plains or plateaus (e.g., Madurai, Trichy), use moderate thresholds (e.g., `low_m: 5`, `high_m: 15`).
"""
    
    combined = "\n\n".join(context_parts)
    if combined:
        combined += "\n\n" + geothink_docs
    else:
        combined = geothink_docs

    print(f"[RAG] Total context: {len(combined)} chars")
    return combined