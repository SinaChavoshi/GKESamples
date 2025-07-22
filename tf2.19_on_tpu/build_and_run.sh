#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export IMAGE_NAME="tf-debug-gke" # Updated image name
export YAML_FILE="tf_debug_job.yaml"

# Color definitions for echo statements
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ----------- Infrastructure Setup (Run Once) ----------- ##
echo -e "${ORANGE}🔧 Configuring Docker and ensuring Artifact Registry repo exists...${NC}"
gcloud auth configure-docker ${REGION}-docker.pkg.dev
if ! gcloud artifacts repositories describe ${AR_REPO_NAME} --project=${PROJECT_ID} --location=${REGION} &> /dev/null; then
  echo -e "${ORANGE}🎨 Artifact Registry repository '${AR_REPO_NAME}' not found. Creating it...${NC}"
  gcloud artifacts repositories create ${AR_REPO_NAME} --repository-format=docker --location=${REGION} --project=${PROJECT_ID}
else
  echo -e "${GREEN}✅ Artifact Registry repository '${AR_REPO_NAME}' already exists.${NC}"
fi

## -------------- Prerequisite Checks ----------------- ##
echo -e "${ORANGE}🕵️ Checking for JobSet CRD and Service Account...${NC}"
# Check for and install JobSet CRD
if ! kubectl get crd jobsets.jobset.x-k8s.io &> /dev/null; then
  echo -e "${ORANGE}🚀 JobSet CRD not found. Installing...${NC}"
  kubectl apply --server-side --force-conflicts -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.6.0/manifests.yaml
else
  echo -e "${GREEN}✅ JobSet CRD is already installed.${NC}"
fi

# Check for and create the Service Account
if ! kubectl get sa tpu-training-sa &> /dev/null; then
  echo -e "${ORANGE}👤 Service Account 'tpu-training-sa' not found. Creating it...${NC}"
  kubectl create sa tpu-training-sa
else
  echo -e "${GREEN}✅ Service Account 'tpu-training-sa' already exists.${NC}"
fi

## ----------- Docker Build & Push ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the Docker image...${NC}"
DOCKER_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
docker build -f Dockerfile -t ${DOCKER_IMAGE_URL} .
docker push ${DOCKER_IMAGE_URL}
echo -e "${GREEN}✅ Docker image pushed successfully to ${DOCKER_IMAGE_URL}${NC}"

## ------------- Job Deployment, Cleanup, and Logging -------------- ##
JOBSET_NAME=$(grep 'name:' ${YAML_FILE} | head -n 1 | awk '{print $2}')
echo -e "${ORANGE}▶️  Starting process for JobSet: ${JOBSET_NAME}${NC}"

echo -e "${ORANGE}🧹 Cleaning up any pre-existing JobSet '${JOBSET_NAME}'...${NC}"
kubectl delete jobset ${JOBSET_NAME} --ignore-not-found=true
# Wait for deletion to complete to avoid conflicts.
echo -e "${ORANGE}⏳ Waiting for resources to be fully deleted...${NC}"
# This is a simple wait; a more robust solution might poll until the jobset is gone.
sleep 15

echo -e "${ORANGE}🚢 Deploying JobSet from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ JobSet '${JOBSET_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for controller pod to be created...${NC}"
CONTROLLER_POD=""
TIMEOUT=300 # 5 minutes timeout
COUNTER=0
INTERVAL=10

# Poll until the controller pod is found or a timeout is reached.
while [ -z "$CONTROLLER_POD" ]; do
    # Attempt to get the pod name. Redirect stderr to /dev/null to suppress errors when the pod is not yet found.
    # The '|| true' ensures the script doesn't exit if the command fails.
    CONTROLLER_POD=$(kubectl get pods -l jobset.sigs.k8s.io/job-name=${JOBSET_NAME}-tpu-controller-0 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    
    if [ -z "$CONTROLLER_POD" ]; then
        if [ $COUNTER -ge $TIMEOUT ]; then
            echo -e "${ORANGE}🛑 TIMEOUT: Controller pod was not found after ${TIMEOUT} seconds.${NC}"
            echo -e "${ORANGE}🔎 Final pod status check:${NC}"
            kubectl get pods -o wide
            exit 1
        fi
        echo "Still waiting for controller pod... (${COUNTER}s / ${TIMEOUT}s)"
        sleep $INTERVAL
        COUNTER=$((COUNTER + INTERVAL))
    else
        echo -e "${GREEN}✅ Found controller pod: ${CONTROLLER_POD}${NC}"
    fi
done

echo -e "${ORANGE}📝 Waiting for pod '${CONTROLLER_POD}' to be ready...${NC}"
kubectl wait --for=condition=Ready pod/${CONTROLLER_POD} --timeout=300s

echo -e "${ORANGE}🪵 Tailing logs for the 'tpu-controller' pod. Press Ctrl+C to stop.${NC}"
kubectl logs -f ${CONTROLLER_POD}

echo -e "${GREEN}✅ Script finished.${NC}"

