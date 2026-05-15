import os
import logging
from typing import Any

# Render startup: populate the service account file from an env var before Vertex AI / Google SDK is initialized.
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if GOOGLE_APPLICATION_CREDENTIALS_JSON:
    _GCP_KEY_PATH = "/tmp/gcp-key.json"
    try:
        with open(_GCP_KEY_PATH, "w", encoding="utf-8") as _f:
            _f.write(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _GCP_KEY_PATH
    except OSError as _exc:
        raise RuntimeError(f"Unable to write Google credentials to {_GCP_KEY_PATH}") from _exc

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stock_guide_agent.agent import root_agent

app = FastAPI(
    title="Stock Guide Agent",
    description="A deployable FastAPI wrapper around the Google ADK stock guide agent.",
    version="1.0.0",
)

logger = logging.getLogger("stock_guide_agent")
logging.basicConfig(level=logging.INFO)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def _call_root_agent(question: str) -> Any:
    if hasattr(root_agent, "run"):
        return root_agent.run(question)
    if callable(root_agent):
        return root_agent(question)
    if hasattr(root_agent, "execute"):
        return root_agent.execute(question)
    raise RuntimeError("Root agent does not expose a supported execution method.")


def _normalize_agent_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("answer", "output", "response", "message", "result"):
            if key in response and isinstance(response[key], str):
                return response[key]
        return str(response)
    return str(response)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="The question field must not be empty.")

    try:
        logger.info("Received question for root_agent: %s", request.question)
        result = _call_root_agent(request.question.strip())
        answer = _normalize_agent_response(result)
        return AskResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to execute root_agent")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request."
        ) from exc


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
