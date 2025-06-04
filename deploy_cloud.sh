#!/bin/bash
set -eu

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPO_NAME=mini-search-engine
TAG=mini-search-engine-api
IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${TAG}:latest

# Vespa Cloud configuration (update these with your actual values)
VESPA_URL=${VESPA_URL:-"https://your-vespa-cloud-endpoint.com"}

echo "🚀 Deploying Mini Search Engine to Google Cloud Run"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Image: ${IMAGE_URI}"

# One-time authentication and service enablement
echo "🔐 Configuring authentication and enabling services..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "🔍 Checking if repository '${REPO_NAME}' exists in '${REGION}'..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
    echo "📦 Repository not found — creating it now."
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Images for mini search engine API service"
else
    echo "✅ Repository already exists."
fi

# Build and push Docker image
echo "🔨 Building and pushing Docker image..."
docker buildx create --use --name mini_search_builder >/dev/null 2>&1 || true
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --tag ${IMAGE_URI} \
    --push .

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy ${TAG} \
    --image ${IMAGE_URI} \
    --region ${REGION} \
    --cpu 2 \
    --memory 4Gi \
    --cpu-boost \
    --no-cpu-throttling \
    --execution-environment gen2 \
    --min-instances 1 \
    --max-instances 1 \
    --concurrency 20 \
    --timeout 300 \
    --allow-unauthenticated \
    --set-env-vars="VESPA_URL=${VESPA_URL}" \
    --port 8000

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${TAG} --region=${REGION} --format="value(status.url)")

echo "🎉 Deployment complete!"
echo "   Service URL: ${SERVICE_URL}"
echo "   API Health: ${SERVICE_URL}/health"
echo "   Search UI: ${SERVICE_URL}/"
echo ""
echo "💡 Next steps:"
echo "   1. Update VESPA_URL environment variable with your actual Vespa Cloud endpoint"
echo "   2. Configure Vespa Cloud authentication if needed"
echo "   3. Test the deployment: curl ${SERVICE_URL}/health"
echo "   4. Index your data to Vespa Cloud using: poetry run python indexer/indexer.py"