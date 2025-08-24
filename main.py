from dotenv import load_dotenv
import os
from google import genai
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from semantic_router.routers import SemanticRouter
from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from typing import List, Optional, Any, Dict, AsyncGenerator, ClassVar
import logging
from contextlib import asynccontextmanager

# --- Setup Logging ---
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(LOG_LEVEL)


# --- Custom LLM Wrapper for Semantic Router ---
class GoogleLLM(BaseLLM):
    """Custom LLM wrapper for Google's GenAI SDK to integrate with semantic-router."""

    _client: ClassVar[Optional[genai.Client]] = None
    kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_client(cls) -> genai.Client:
        """Initializes and returns a singleton GenAI client."""
        if cls._client is None:
            logger.info("Initializing singleton GenAI client...")
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
            cls._client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
        return cls._client

    def __init__(self, name: str, **kwargs: Any):
        """Initializes the GoogleLLM."""
        super().__init__(name=name, **kwargs)
        # The client is lazily loaded on first use via get_client().

    async def __acall__(self, messages: List[Message], **kwargs) -> Optional[str]:
        """Call the Google Gemini model with the provided messages and configuration.

        Args:
            messages: List of messages to send to the model
            **kwargs: Additional arguments including multimodal_contents and config

        Returns:
            The response text from the model or None if no response

        Raises:
            Exception: If the LLM call fails
        """
        logger.info(f"--- Calling Gemini Model: {self.name} ---")

        contents_for_llm = kwargs.get("multimodal_contents", messages[0].content)

        # Get config from kwargs, fallback to instance's default kwargs
        generation_config = kwargs.get("config", self.kwargs.get("config", {}))

        try:
            response = await self.get_client().aio.models.generate_content(
                model=self.name,
                contents=contents_for_llm,
                **generation_config,  # Use the passed-in config
            )
            return response.text if response and hasattr(response, "text") else ""
        except Exception as e:
            logger.error(
                f"Error calling Gemini model '{self.name}': {e}", exc_info=True
            )
            raise Exception(f"LLM call failed for model '{self.name}'.") from e


# --- FastAPI Lifespan Manager for Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize application services during startup."""
    # Configure environment for Google services before any other setup.
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set.")
    os.environ["GOOGLE_PROJECT_ID"] = project_id

    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not location:
        location = "us-central1"
        logger.info("GOOGLE_CLOUD_LOCATION not set, defaulting to 'us-central1'.")
    os.environ["GOOGLE_LOCATION"] = location

    logger.info("Loading semantic router...")
    try:
        router = SemanticRouter.from_yaml("router.yaml")
        if not router.routes:
            raise ValueError("No routes defined in 'router.yaml'.")

        router.sync("local")
        logger.info("Semantic Router index has been successfully populated.")

        default_route_name = router.routes[0].name if router.routes else ""
        app.state.router = router
        app.state.default_route_name = default_route_name
        logger.info(
            f"Semantic router is ready. Fallback route is '{app.state.default_route_name}'."
        )

    except (ValueError, FileNotFoundError) as e:
        logger.error(
            f"FATAL: Application startup failed due to configuration error: {e}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            f"FATAL: An unexpected error occurred during application startup: {e}",
            exc_info=True,
        )
        raise

    yield
    logger.info("Application shutdown.")


# --- FastAPI Application ---
app = FastAPI(
    title="Gemini Model Router",
    description="A router for directing queries to different Gemini models based on semantic meaning.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Request model for querying the router with content and configuration."""

    contents: List[Dict[str, Any]] = Field(
        description=(
            "A list of content parts (e.g., text, image) to be processed. "
            "Follows the structure of `genai.types.ContentDict`."
        )
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through config for `genai.types.GenerateContentConfig`.",
    )


class RouterResponse(BaseModel):
    """Response model containing the route name and model response."""

    route_name: str
    model_response: str


def _get_text_for_routing(contents: List[Dict[str, Any]]) -> str:
    """Extracts the first text part from the contents for routing."""
    text = next((part["text"] for part in contents if "text" in part), "")
    if not text:
        raise HTTPException(
            status_code=400, detail="No text found in query for routing."
        )
    logger.info(f"Routing text: '{text}'")
    return text


def _determine_route(
    router: SemanticRouter, text: str, default_route: str
) -> RouteChoice:
    """Determines the route for the given text."""
    route_result = router(text)
    logger.debug(f"Raw route_result from router: {route_result}")

    # Handle both single RouteChoice and List[RouteChoice]
    if isinstance(route_result, list):
        route_choice = route_result[0] if route_result else None
    else:
        route_choice = route_result

    if route_choice and route_choice.name:
        score = route_choice.similarity_score
        score_str = f"{score:.4f}" if score is not None else "N/A"
        logger.info(f"  - Route Choice: {route_choice.name}, Similarity: {score_str}")
        return route_choice
    else:
        logger.info(f"No specific route found. Using fallback: '{default_route}'.")
        # Create a mock RouteChoice for the default route
        return RouteChoice(name=default_route)


async def _execute_llm_call(
    route: Any, contents: List[Dict[str, Any]], config: Dict[str, Any], text: str
) -> str:
    """Executes the call to the selected LLM."""
    if not route.llm:
        raise HTTPException(
            status_code=500,
            detail=f"Route '{route.name}' is defined but has no configured LLM.",
        )

    llm_instance = route.llm
    if not isinstance(llm_instance, GoogleLLM):
        raise HTTPException(
            status_code=500,
            detail=f"LLM for route '{route.name}' is not a GoogleLLM instance.",
        )

    messages = [Message(role="user", content=text)]
    response_text = await llm_instance.__acall__(
        messages, multimodal_contents=contents, config=config
    )
    return response_text or ""


@app.post(
    "/query",
    response_model=RouterResponse,
    summary="Route a query (text-only or multimodal)",
)
async def handle_query(request: QueryRequest, fastapi_request: Request):
    """
    Routes queries by analyzing the first text part of the `contents` payload.

    For multimodal queries (e.g., text + image), routing is based solely on the
    text. The complete, unprocessed `contents` are then passed to the selected
    downstream model.

    Args:
        request: The query request containing contents and config
        fastapi_request: The FastAPI request object

    Returns:
        RouterResponse: The response containing the route name and model response

    Raises:
        HTTPException: If no text is found in query or if routing fails
    """
    router = fastapi_request.app.state.router
    default_route_name = fastapi_request.app.state.default_route_name

    try:
        text_for_routing = _get_text_for_routing(request.contents)
        route_choice = _determine_route(router, text_for_routing, default_route_name)
        chosen_route = router.get(route_choice.name)

        if not chosen_route:
            raise HTTPException(
                status_code=500,
                detail=f"Route '{route_choice.name}' could not be found.",
            )
        logger.info(f"Query routed to: {chosen_route.name}")

        model_response = await _execute_llm_call(
            chosen_route, request.contents, request.config, text_for_routing
        )

        return RouterResponse(
            route_name=chosen_route.name, model_response=model_response
        )

    except HTTPException as http_exc:
        logger.warning(f"HTTP Exception in /query: {http_exc.detail}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in /query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected internal server error occurred: {e}"
        )


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""

    status: str
    detail: Optional[str] = None


@app.get(
    "/health/router",
    response_model=HealthCheckResponse,
    summary="Health check for the semantic router",
)
def get_router_health(request: Request) -> HealthCheckResponse:
    """Check the health of the semantic router.

    Returns:
        HealthCheckResponse: The health status of the router
    """
    try:
        router = request.app.state.router
        if router and router.routes:
            return HealthCheckResponse(status="ok")
        return HealthCheckResponse(
            status="error", detail="Router is not initialized or has no routes."
        )
    except AttributeError:
        return HealthCheckResponse(
            status="error", detail="Router not found in application state."
        )


@app.get("/", summary="Health Check")
def read_root() -> Dict[str, str]:
    """Health check endpoint to confirm the service is running.

    Returns:
        Dict[str, str]: A dictionary with status and embedding model information
    """
    return {"status": "ok", "embedding_model": "Vertex AI (via GoogleEncoder)"}


# --- Server Startup (for local testing) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
