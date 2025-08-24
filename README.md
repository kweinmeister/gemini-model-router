# 🤖 Gemini Model Router

A smart, FastAPI-powered router that dynamically routes your queries to the best Gemini model for the job, using the Google GenAI SDK for Vertex AI. 🚀

This project uses `semantic-router` to intelligently decide which Gemini model should handle a given query based on its content. It's a lightweight, efficient, and scalable solution for building powerful multimodal applications, with all routing logic defined in a simple [`router.yaml`](./router.yaml) file.

## ✨ Features

- **🧠 Smart Routing:** Uses [`semantic-router`](https://github.com/aurelio-labs/semantic-router) to direct queries to the most appropriate model based on semantic meaning.
- **⚡️ FastAPI Powered:** Asynchronous, fast, and production-ready API using the latest Google GenAI SDK for Vertex AI.
- **🔧 Centralized Configuration:** Easily configure all routes, models, and example utterances in a single [`router.yaml`](./router.yaml) file.

## 🧠 Routing Logic

The routing is determined by semantic similarity, as defined in [`router.yaml`](./router.yaml). The router matches the user's query to the best route based on example utterances.

- **`gemini-2.5-pro`**: For complex, multi-step tasks requiring deep reasoning, code generation, and analysis of large documents.

  - **Recommended for:** In-depth analysis, comprehensive business plans, full-stack application development.

- **`gemini-2.5-flash`**: A balanced model for tasks that require a mix of speed, cost-efficiency, and strong reasoning capabilities.

  - **Recommended for:** Summarization, brainstorming, professional communication, and content explanation.

- **`gemini-2.5-flash-lite`**: The fastest and most cost-effective model, optimized for high-volume, low-latency tasks like classification and data extraction.

  - **Recommended for:** Classification, translation, data extraction, and quick conversational queries.

## 🚀 Getting Started

### Prerequisites

1. A Google Cloud Project with billing enabled.
1. The `gcloud` CLI installed and authenticated (`gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`).
1. The following Google Cloud APIs enabled: Cloud Run, Vertex AI, Cloud Build, and Artifact Registry.

### ⚙️ Configuration

The application is configured using environment variables. For both local development and deployment, set the following variables in your shell.

```sh
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export SERVICE_NAME="gemini-model-router"
export DEBUG="False"  # Optional: Set to "True" for verbose logging
```

For local development, the application will automatically load these variables if you place them in a `.env` file in the project root.

### Installation & Local Run

1. **Clone the repository:**

   ```sh
   git clone http://github.com/kweinmeister/gemini-model-router
   cd gemini-model-router
   ```

1. **Create and activate a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate
   ```

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   For development (including testing), use:

   ```sh
   pip install -r requirements-dev.txt
   ```

1. **Set up your environment:**
   Set the environment variables as described in the [Configuration](#%EF%B8%8F-configuration) section (either by exporting them or using a `.env` file).

1. **Run the application locally:**

   ```sh
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8080`.

## 📖 API Endpoints

The application exposes the following endpoints:

- **`POST /query`**: The primary endpoint for routing multimodal queries. It expects a JSON payload with `contents` and an optional `config` object.
- **`GET /health/router`**: A health check endpoint that verifies whether the semantic router is initialized and has routes loaded. Returns `{"status": "ok"}` on success.
- **`GET /`**: A basic health check endpoint to confirm the service is running.

## ☁️ Deployment to Cloud Run

1. **Deploy Directly from Source Code**:
   First, ensure the environment variables are set as described in the [Configuration](#%EF%B8%8F-configuration) section. Then, run the following command to deploy.

   ```sh
   gcloud run deploy "$SERVICE_NAME" \
     --source . \
     --region="$GOOGLE_CLOUD_LOCATION" \
     --allow-unauthenticated \
     --labels="dev-tutorial=code-model-router" \
     --set-env-vars="GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT" \
     --set-env-vars="GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
   ```

## 🔐 Security Considerations

**Warning:** The default deployment command uses the `--allow-unauthenticated` flag, which makes your Cloud Run service publicly accessible. This is convenient for testing but is not recommended for production environments.

For production, you should remove this flag and implement authentication. For additional considerations, including input sanitization and other security enhancements, see the [Future Enhancements](#future-enhancements) section.

Here are some recommended authentication methods:

- **Identity-Aware Proxy (IAP):** Secure your application with Google Cloud's IAP to control access.
- **API Gateway:** Place your service behind an API Gateway to manage authentication, rate limiting, and monitoring.
- **Service-to-Service Authentication:** If your service will be called by other services, use IAM-based service accounts for secure, authenticated communication.

Always ensure your production deployments are secured to prevent unauthorized access and potential misuse.

## 🧪 Testing the Endpoint

To test the endpoint, first set the `SERVICE_URL` environment variable. The `curl` commands below will work for both local and deployed environments.

### 1. Set the Service URL

- **For local testing:**

  ```sh
  export SERVICE_URL="http://127.0.0.1:8080"
  ```

- **For a deployed Cloud Run service:**
  This command retrieves your service URL and exports it as an environment variable.

  ```sh
  export SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format="value(status.url)")
  echo "Service URL set to: $SERVICE_URL"
  ```

### 2. Send Test Queries

Now you can use `curl` to test the different routing tiers.

- **Test a complex query (routes to `gemini-2.5-pro`):**

  ```sh
  curl -X POST "$SERVICE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"text": "Compare and contrast the philosophical implications of determinism and free will in the context of advanced artificial intelligence, citing relevant academic sources."}]}'
  ```

- **Test a general-purpose query (routes to `gemini-2.5-flash`):**

  ```sh
  curl -X POST "$SERVICE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"text": "Help me brainstorm some creative ideas for a team-building event for a remote-first company."}]}'
  ```

- **Test a simple query (routes to `gemini-2.5-flash-lite`):**

  ```sh
  curl -X POST "$SERVICE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"text": "Translate ''How much does this cost?'' to Japanese."}]}'
  ```

- **Check the router's health status:**

```sh
curl "$SERVICE_URL/health/router"
```

## 📜 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

<a name="future-enhancements"></a>

## 🌟 Future Enhancements

To make this project more robust for production environments, the following enhancements should be considered.

### Input Validation and Sanitization

- **Character Limit Enforcement:** Add a configurable maximum input length to prevent overly long (and costly) queries.
- **Malicious Character Filtering:** Implement character stripping or escaping to protect against prompt injection.
- **Content Moderation:** Integrate with a content moderation API to filter potentially harmful content.

### Robustness and Reliability

- **Model Fallback Mechanism:** Implement automatic fallback to an alternative model if a primary model call fails.
- **Retry Logic:** Add a retry mechanism for transient network errors or model failures.
- **Rate Limiting:** Implement request rate limiting to manage costs and prevent abuse.

### Security Enhancements

- **Authentication:** Secure the API with a robust authentication mechanism for production use.
- **Authorization:** Implement user or session-based routing and model access controls.
- **Request Logging:** Add comprehensive logging for security monitoring and auditing purposes.

### Performance and Monitoring

- **Caching:** Implement caching for frequently requested routes or identical queries to improve latency and reduce costs.
- **Metrics Collection:** Add metrics collection to monitor performance, usage patterns, and model accuracy.
- **Expanded Health Checks:** Enhance the health check endpoint to include model availability and performance metrics, providing a more comprehensive status of the service.
