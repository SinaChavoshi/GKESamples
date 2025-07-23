#!/bin/bash
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="tpu-vm-gke-testing"
export CLUSTER_REGION="us-central2"
export CLUSTER_NAME="chavoshi-test"
export AR_REGION="us-central2"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"

export MASTER_IMAGE_NAME="dlrm-master"
export WORKER_IMAGE_NAME="dlrm-worker"
export YAML_FILE="tfjob_checkpoint_repro.yaml"
export TFJOB_NAME="tf-16-dlrm-tfjob"

# Color definitions
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ----------- Authenticate & Configure ----------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${CLUSTER_REGION} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker for region ${AR_REGION}...${NC}"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

## ----------- Docker Build & Push ----------- ##
echo -e "${ORANGE}🚀 Building Master image...${NC}"
MASTER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${MASTER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f master.Dockerfile -t ${MASTER_IMAGE_URL} .
docker push ${MASTER_IMAGE_URL}
echo -e "${GREEN}✅ Master image pushed.${NC}"

echo -e "${ORANGE}🚀 Building Worker image...${NC}"
WORKER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${WORKER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f worker.Dockerfile -t ${WORKER_IMAGE_URL} .
docker push ${WORKER_IMAGE_URL}
echo -e "${GREEN}✅ Worker image pushed.${NC}"

## ------------- TFJob Deployment & Logging ------------- ##
echo -e "${ORANGE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"
echo -e "${ORANGE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} -n default --ignore-not-found=true
sleep 5

echo -e "${ORANGE}🚢 Deploying TFJob from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=15m

echo -e "${ORANGE}🪵 Tailing logs for the MASTER pod. Training output appears here. Press Ctrl+C to stop.${NC}"
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}

echo -e "${GREEN}✅ Script finished.${NC}"
