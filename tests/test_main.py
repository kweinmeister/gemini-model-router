import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from main import app, GoogleLLM
from semantic_router.schema import Message, RouteChoice


@pytest.fixture
def client(mocker):
    """Provide a test client with a mocked router."""
    # Mock the router and its routes before the app starts
    mocker.patch("main.SemanticRouter.from_yaml", return_value=MagicMock())
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_google_llm_wrapper_acall(mocker):
    """Test that GoogleLLM wrapper correctly calls the underlying client."""
    # Mock the get_client class method
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    mocker.patch.object(GoogleLLM, "get_client", return_value=mock_client)

    # Create GoogleLLM instance (client is now managed internally)
    llm = GoogleLLM(name="gemini-test")
    test_messages = [Message(role="user", content="test query")]

    # Call the LLM
    await llm.__acall__(test_messages)

    # Verify the underlying client was called
    mock_client.aio.models.generate_content.assert_called_once()


def test_google_llm_initialization(mocker):
    """Test GoogleLLM initialization with different parameter combinations."""
    # Mock the get_client class method
    mocker.patch.object(GoogleLLM, "get_client", return_value=MagicMock())

    # Test basic initialization
    llm = GoogleLLM(name="test-model")
    assert llm.name == "test-model"

    # Test initialization with additional parameters
    llm_with_params = GoogleLLM(name="test-model-2", temperature=0.7, max_tokens=100)
    assert llm_with_params.name == "test-model-2"
    assert llm_with_params.temperature == 0.7
    assert llm_with_params.max_tokens == 100


@pytest.mark.asyncio
async def test_google_llm_wrapper_acall_error_handling(mocker):
    """Test error handling in GoogleLLM wrapper's async call."""
    # Mock the client to raise an exception
    mock_client = MagicMock()
    mock_client.aio.models.generate_content.side_effect = Exception("API Error")
    mocker.patch.object(GoogleLLM, "get_client", return_value=mock_client)

    llm = GoogleLLM(name="gemini-test")
    test_messages = [Message(role="user", content="test query")]

    # Verify the exception is properly raised and wrapped
    with pytest.raises(Exception, match="LLM call failed for model 'gemini-test'"):
        await llm.__acall__(test_messages)


@pytest.mark.asyncio
async def test_lifespan_function(mocker):
    """Test the simplified lifespan function for proper initialization."""
    # Mock dependencies
    mock_semantic_router = mocker.patch("main.SemanticRouter")
    mock_router = MagicMock()
    mock_router.routes = [MagicMock()]  # Ensure routes list is not empty
    mock_semantic_router.from_yaml.return_value = mock_router

    # Mock environment variables
    mocker.patch(
        "os.getenv",
        side_effect=lambda key, default=None: {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "test-location",
        }.get(key, default),
    )

    # Create a mock app
    mock_app = MagicMock()
    mock_app.state = MagicMock()

    # Test the lifespan function
    from main import lifespan

    # Get the async generator
    lifespan_gen = lifespan(mock_app)

    # Start the lifespan context
    await lifespan_gen.__aenter__()

    # Verify initialization
    mock_semantic_router.from_yaml.assert_called_once_with("router.yaml")
    mock_router.sync.assert_called_once_with("local")
    assert hasattr(mock_app.state, "router")
    assert hasattr(mock_app.state, "default_route_name")

    # Clean up the lifespan context
    await lifespan_gen.__aexit__(None, None, None)


def test_handle_query_edge_cases(client, mocker):
    """Test edge cases in the handle_query function."""
    # Mock router setup
    mock_router = MagicMock()
    mock_router.return_value = RouteChoice(name="test-route")

    # Mock a route with no LLM
    mock_route = MagicMock()
    mock_route.llm = None
    mock_route.name = "test-route"
    mock_router.get.return_value = mock_route

    mocker.patch.object(client.app.state, "router", mock_router)

    # Test route with no LLM configured
    request_payload = {"contents": [{"text": "test query"}], "config": {}}
    response = client.post("/query", json=request_payload)
    assert response.status_code == 500
    assert "is defined but has no configured LLM" in response.json()["detail"]

    # Test route with non-GoogleLLM LLM
    mock_route.llm = MagicMock()  # Not a GoogleLLM instance
    response2 = client.post("/query", json=request_payload)
    assert response2.status_code == 500
    assert "is not a GoogleLLM instance" in response2.json()["detail"]


@pytest.mark.parametrize(
    "query, mock_route_choice, expected_route",
    [
        (
            "test utterance for route 1",
            RouteChoice(name="test-route-1"),
            "test-route-1",
        ),
        (
            "test utterance for route 2",
            RouteChoice(name="test-route-2"),
            "test-route-2",
        ),
        ("An unknown query", None, "test-route-1"),  # Fallback case
    ],
)
def test_routing_logic(query, mock_route_choice, expected_route, client, mocker):
    """Test that queries are routed correctly based on content."""
    # Setup mock router
    mock_router = MagicMock()
    mock_router.return_value = mock_route_choice

    # Mock route with LLM
    mock_llm = AsyncMock(spec=GoogleLLM)
    mock_llm.__acall__.return_value = "mocked LLM response"
    mock_route = MagicMock()
    mock_route.llm = mock_llm
    mock_route.name = expected_route
    mock_router.get.return_value = mock_route

    mocker.patch.object(client.app.state, "router", mock_router)
    mocker.patch.object(client.app.state, "default_route_name", "test-route-1")
    mocker.patch("asyncio.to_thread", return_value="mocked LLM response")

    # Process query
    response = client.post("/query", json={"contents": [{"text": query}]})

    # Verify routing
    assert response.status_code == 200
    assert response.json()["route_name"] == expected_route


def test_query_endpoint_no_text(client):
    """Test that a 400 error is returned when no text is provided for routing."""
    # Request with no text part
    request_payload = {"contents": [{"image": "some_image_data"}]}

    # Call endpoint
    response = client.post("/query", json=request_payload)

    # Verify error response
    assert response.status_code == 400
    assert "No text found in query for routing." in response.json()["detail"]


# --- Integration Tests ---


def test_query_endpoint_success(client, mocker):
    """Test the /query endpoint with a valid request."""
    # Setup mock router
    mock_router = MagicMock()
    mock_router.return_value = RouteChoice(name="test-route-1")

    # Mock route with LLM
    mock_llm = AsyncMock(spec=GoogleLLM)
    mock_llm.__acall__.return_value = "mocked LLM response"
    mock_route = MagicMock()
    mock_route.llm = mock_llm
    mock_route.name = "test-route-1"
    mock_router.get.return_value = mock_route

    mocker.patch.object(client.app.state, "router", mock_router)
    mocker.patch("asyncio.to_thread", return_value="mocked LLM response")

    # Valid request
    request_payload = {
        "contents": [{"text": "This is a test utterance for route 1"}],
        "config": {"temperature": 0.7},
    }

    # Call endpoint
    response = client.post("/query", json=request_payload)

    # Verify successful response
    assert response.status_code == 200

    # Verify response structure
    response_data = response.json()
    assert "route_name" in response_data
    assert "model_response" in response_data
    assert response_data["route_name"] == "test-route-1"
    assert response_data["model_response"] == "mocked LLM response"


def test_health_check_endpoint(client):
    """Test the health check endpoint."""
    # Call health check endpoint
    response = client.get("/")

    # Verify response
    assert response.status_code == 200
    expected_response = {
        "status": "ok",
        "embedding_model": "Vertex AI (via GoogleEncoder)",
    }
    assert response.json() == expected_response


def test_router_health_check_endpoint(client, mocker):
    """Test the router health check endpoint."""
    # Test healthy router
    mock_router = MagicMock()
    mock_router.routes = [MagicMock()]  # Simulate having at least one route
    mocker.patch.object(client.app.state, "router", mock_router)

    response = client.get("/health/router")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "detail": None}

    # Test router with no routes
    mock_router.routes = []
    response_no_routes = client.get("/health/router")
    assert response_no_routes.status_code == 200
    assert response_no_routes.json() == {
        "status": "error",
        "detail": "Router is not initialized or has no routes.",
    }

    # Test missing router attribute
    original_router = client.app.state.router
    try:
        del client.app.state.router
        response_no_router_attr = client.get("/health/router")
        assert response_no_router_attr.status_code == 200
        assert response_no_router_attr.json() == {
            "status": "error",
            "detail": "Router not found in application state.",
        }
    finally:
        # Restore the state to prevent teardown errors
        client.app.state.router = original_router
