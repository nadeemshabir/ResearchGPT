#!/usr/bin/env bash
#
# Deploy ResearchGPT to Google Cloud Run.
#
#   ./deploy/cloudrun.sh PROJECT_ID [REGION]
#
# Builds the image with Cloud Build, stores it in Artifact Registry, and
# deploys a service. Re-running it deploys a new revision; traffic moves only
# once the new revision passes its health check.
#
# The API key is read from Secret Manager rather than passed as an environment
# variable, so it never appears in `gcloud run services describe` output or in
# the Cloud Console UI.

set -euo pipefail

PROJECT="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="researchgpt"
REPO="researchgpt"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}"

echo "==> Project ${PROJECT}, region ${REGION}"
gcloud config set project "${PROJECT}" --quiet

echo "==> Enabling APIs (no-op if already on)"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    --quiet

echo "==> Ensuring the Artifact Registry repository exists"
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" --quiet \
    >/dev/null 2>&1 \
    || gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="ResearchGPT container images" \
        --quiet

echo "==> Building. The corpus is fetched and embedded inside the build, which"
echo "    is the point: Cloud Run's filesystem is ephemeral, so the vector"
echo "    store has to ship in the image."
#
# Machine type matters to the bill, not just the clock. Cloud Build's free
# allowance covers the *default* machine only; asking for e2-highcpu-8 is
# billed from the first minute. The default is slower (roughly 25-35 minutes
# against 15) and free, so it is what runs unless BUILD_MACHINE says otherwise:
#
#   BUILD_MACHINE=e2-highcpu-8 ./deploy/cloudrun.sh PROJECT
#
BUILD_MACHINE="${BUILD_MACHINE:-}"
BUILD_ARGS=(--tag "${IMAGE}" --timeout=45m)
if [[ -n "${BUILD_MACHINE}" ]]; then
    echo "    Using ${BUILD_MACHINE} -- faster, and billed."
    BUILD_ARGS+=(--machine-type="${BUILD_MACHINE}")
else
    echo "    Using the default machine (slower, covered by the free tier)."
fi

gcloud builds submit "${BUILD_ARGS[@]}" .

echo "==> Deploying"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    `# torch plus the embedding model sit around 1.5GB resident. 2Gi leaves` \
    `# room for a few concurrent requests without the kernel killing us.` \
    --memory 2Gi \
    --cpu 2 \
    `# Startup loads the model and builds a BM25 index over every chunk.` \
    `# Cloud Run's default 240s startup budget is enough, but only just on a` \
    `# cold image pull, so it is raised.` \
    --timeout 300 \
    `# The app answers requests on a thread pool sized from the CPU count.` \
    `# Cloud Run's default of 80 concurrent requests per instance would queue` \
    `# far more work than one container can do, turning a busy moment into` \
    `# timeouts rather than a second instance.` \
    --concurrency 8 \
    --max-instances 3 \
    `# Scale to zero. This is what keeps it inside the free tier, and it is` \
    `# why the first visitor after an idle period waits for a cold start.` \
    --min-instances 0 \
    --set-env-vars "LLM_PROVIDER=gemini,LLM_MODEL=gemini-2.5-flash" \
    --set-secrets "GEMINI_API_KEY=researchgpt-gemini-key:latest" \
    --quiet

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"

echo
echo "==> Deployed: ${URL}"
echo "==> Verifying"
curl -fsS --max-time 120 "${URL}/health" | python -m json.tool
