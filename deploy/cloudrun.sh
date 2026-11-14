#!/usr/bin/env bash
#
# Deploy ResearchGPT to Google Cloud Run.
#
#   UPLOAD_BUCKET=my-bucket ./deploy/cloudrun.sh PROJECT_ID [REGION]
#
# Builds the image with Cloud Build, stores it in Artifact Registry, and
# deploys a service. Re-running it deploys a new revision; traffic moves only
# once the new revision passes its health check.
#
# Secrets are read from Secret Manager rather than passed as environment
# variables, so they never appear in `gcloud run services describe` output or
# in the Cloud Console UI.
#
# The service starts with an EMPTY knowledge base. It has no corpus of its own
# and ships no papers; everything it can answer from is uploaded by its users.
# That needs two things to outlive the container, because Cloud Run's
# filesystem does not survive a restart and is not shared between instances:
#
#   UPLOAD_BUCKET  a GCS bucket for the uploaded PDFs        (required)
#   a vector store reachable over the network                (required)
#
# One-time setup for both is in docs/DEPLOY.md. This script refuses to deploy
# without them rather than producing a service that loses every upload.

set -euo pipefail

PROJECT="${1:?Usage: UPLOAD_BUCKET=... $0 PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="researchgpt"
REPO="researchgpt"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}"

# Storage for uploaded PDFs.
UPLOAD_BUCKET="${UPLOAD_BUCKET:?Set UPLOAD_BUCKET to the GCS bucket for uploaded PDFs. See docs/DEPLOY.md}"
UPLOAD_PREFIX="${UPLOAD_PREFIX:-papers}"

# Vector store. cloud = Chroma Cloud; http = a Chroma server you run.
CHROMA_MODE="${CHROMA_MODE:-cloud}"
# Secret Manager secret holding the Chroma Cloud API key.
CHROMA_KEY_SECRET="${CHROMA_KEY_SECRET:-researchgpt-chroma-key}"

case "${CHROMA_MODE}" in
    cloud)
        CHROMA_TENANT="${CHROMA_TENANT:?CHROMA_MODE=cloud needs CHROMA_TENANT (Chroma Cloud dashboard)}"
        CHROMA_DATABASE="${CHROMA_DATABASE:?CHROMA_MODE=cloud needs CHROMA_DATABASE (Chroma Cloud dashboard)}"
        STORE_ENV="CHROMA_MODE=cloud,CHROMA_TENANT=${CHROMA_TENANT},CHROMA_DATABASE=${CHROMA_DATABASE}"
        STORE_SECRETS=",CHROMA_API_KEY=${CHROMA_KEY_SECRET}:latest"
        ;;
    http)
        CHROMA_HOST="${CHROMA_HOST:?CHROMA_MODE=http needs CHROMA_HOST}"
        STORE_ENV="CHROMA_MODE=http,CHROMA_HOST=${CHROMA_HOST},CHROMA_PORT=${CHROMA_PORT:-8000},CHROMA_SSL=${CHROMA_SSL:-false}"
        STORE_SECRETS=""
        ;;
    *)
        echo "CHROMA_MODE must be 'cloud' or 'http'. 'local' is an on-disk store;" >&2
        echo "on Cloud Run it is wiped on every restart and not shared between" >&2
        echo "instances, so every upload would be lost. See docs/DEPLOY.md." >&2
        exit 1
        ;;
esac

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

echo "==> Checking the upload bucket exists"
gcloud storage buckets describe "gs://${UPLOAD_BUCKET}" --quiet >/dev/null 2>&1 || {
    echo "No bucket gs://${UPLOAD_BUCKET}. Create it and grant the service" >&2
    echo "account access -- the commands are in docs/DEPLOY.md." >&2
    exit 1
}

echo "==> Building. No papers are fetched or indexed: the image ships empty"
echo "    and the service answers from what its users upload."
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
    `# Startup loads the model, opens the remote store and builds a BM25` \
    `# index over whatever is in it. Cloud Run's default 240s startup budget` \
    `# is enough, but only just on a cold image pull, so it is raised.` \
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
    --set-env-vars "LLM_PROVIDER=gemini,LLM_MODEL=gemini-2.5-flash,UPLOAD_BUCKET=${UPLOAD_BUCKET},UPLOAD_PREFIX=${UPLOAD_PREFIX},${STORE_ENV}" \
    --set-secrets "GEMINI_API_KEY=researchgpt-gemini-key:latest${STORE_SECRETS}" \
    --quiet

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"

echo
echo "==> Deployed: ${URL}"
echo "==> Verifying"
curl -fsS --max-time 120 "${URL}/health" | python -m json.tool
echo
echo "A new deployment has no papers. total_papers: 0 above is correct --"
echo "upload one in the UI, or with:"
echo "  curl -F file=@paper.pdf ${URL}/papers"
