#!/bin/bash
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export ZONE="us-east5-a"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"

# New unified configuration
export UNIFIED_IMAGE_NAME="tf-tpu-unified"
export YAML_FILE="tfjob-v6e-16chip.yaml"
export TFJOB_NAME="tf-mnist-v6e-16chip"

# Color definitions
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m'

## ----------- Authenticate & Configure ----------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker...${NC}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

## ----------- Docker Build & Push ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the unified TF image...${NC}"
UNIFIED_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${UNIFIED_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f Dockerfile -t ${UNIFIED_IMAGE_URL} .
docker push ${UNIFIED_IMAGE_URL}
echo -e "${GREEN}✅ Unified image pushed.${NC}"

## ------------- TFJob Deployment & Logging ------------- ##
echo -e "${ORANGE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"

echo -e "${ORANGE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} --ignore-not-found=true

# Wait a few seconds for cleanup to prevent race conditions
sleep 5

echo -e "${ORANGE}🚢 Deploying TFJob from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=10m

echo -e "${ORANGE}🪵 Tailing logs for the MASTER pod. Training output appears here. Press Ctrl+C to stop.${NC}"
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}

echo -e "${GREEN}✅ Script finished.${NC}"
