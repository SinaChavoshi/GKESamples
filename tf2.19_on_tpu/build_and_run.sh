#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5" # The region of your Artifact Registry
export ZONE="us-east5-a"  # The zone of your GKE cluster
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"

# Configuration for the multi-host controller-worker architecture
export WORKER_IMAGE_NAME="tf-tpu-worker"
export CONTROLLER_IMAGE_NAME="tf-tpu-controller"
export YAML_FILE="jobset-tf-v6e.yaml"
export JOBSET_NAME="tf-v6e-multihost-jobset"

# Color definitions for echo statements
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ----------- Authenticate & Configure ----------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker for region ${REGION}...${NC}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
echo -e "${GREEN}✅ Authentication configured.${NC}"

## ----------- Docker Build & Push: Worker ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the gRPC Worker image...${NC}"
WORKER_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${WORKER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f worker.Dockerfile -t ${WORKER_IMAGE_URL} .
docker push ${WORKER_IMAGE_URL}
echo -e "${GREEN}✅ Worker image pushed.${NC}"

## ----------- Docker Build & Push: Controller ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the Training Controller image...${NC}"
CONTROLLER_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${CONTROLLER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f controller.Dockerfile -t ${CONTROLLER_IMAGE_URL} .
docker push ${CONTROLLER_IMAGE_URL}
echo -e "${GREEN}✅ Controller image pushed.${NC}"

## ------------- JobSet Deployment & Logging ------------- ##
echo -e "${ORANGE}▶️  Starting process for JobSet: ${JOBSET_NAME}${NC}"

echo -e "${ORANGE}🧹 Cleaning up any pre-existing JobSet '${JOBSET_NAME}'...${NC}"
kubectl delete jobset ${JOBSET_NAME} --ignore-not-found=true

echo -e "${ORANGE}⏳ Waiting for old resources to be fully deleted...${NC}"
kubectl wait --for=delete jobset/${JOBSET_NAME} --timeout=60s || true
echo -e "${GREEN}✅ Cleanup complete.${NC}"

echo -e "${ORANGE}🚢 Deploying JobSet from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ JobSet '${JOBSET_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the TPU CONTROLLER pod to be created and ready...${NC}"
# Wait for the controller pod to reach the 'Ready' state.
# This ensures the worker pods are likely up and running before we tail logs.
kubectl wait --for=condition=Ready pod -l jobset.sigs.k8s.io/replicated-job-name=tpu-controller --timeout=10m

echo -e "${ORANGE}🪵 Tailing logs for the CONTROLLER pod. Training output will appear here. Press Ctrl+C to stop.${NC}"
kubectl logs -f -l jobset.sigs.k8s.io/replicated-job-name=tpu-controller

echo -e "${GREEN}✅ Script finished.${NC}"
