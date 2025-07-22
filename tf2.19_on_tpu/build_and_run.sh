#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export IMAGE_NAME="tf-debug-gke"
export YAML_FILE="jobset.yaml"
export JOBSET_NAME="tpu-v5e-training-job"

# Color definitions for echo statements
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ----------- Authenticate & Configure ----------- ##
echo -e "${ORANGE}🔐 Configuring gcloud and Docker...${NC}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
echo -e "${GREEN}✅ Authentication configured.${NC}"

## ----------- Docker Build & Push ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the Docker image...${NC}"
DOCKER_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
docker build -f Dockerfile -t ${DOCKER_IMAGE_URL} .
docker push ${DOCKER_IMAGE_URL}
echo -e "${GREEN}✅ Docker image pushed successfully to ${DOCKER_IMAGE_URL}${NC}"

## ------------- Job Deployment, Cleanup, and Logging -------------- ##
echo -e "${ORANGE}▶️  Starting process for JobSet: ${JOBSET_NAME}${NC}"

echo -e "${ORANGE}🧹 Cleaning up any pre-existing JobSet '${JOBSET_NAME}'...${NC}"
kubectl delete jobset ${JOBSET_NAME} --ignore-not-found=true

# **CORRECTION**: Use 'kubectl wait' for deletion instead of 'sleep'. It's more reliable.
# The '|| true' prevents the script from exiting if the jobset was already gone.
echo -e "${ORANGE}⏳ Waiting for old resources to be fully deleted...${NC}"
kubectl wait --for=delete jobset/${JOBSET_NAME} --timeout=60s || true
echo -e "${GREEN}✅ Cleanup complete.${NC}"

echo -e "${ORANGE}🚢 Deploying JobSet from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ JobSet '${JOBSET_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for a worker pod to be created...${NC}"
TARGET_POD=""
TIMEOUT=300 # 5 minutes timeout
COUNTER=0
INTERVAL=10

# Poll until the target pod is found or a timeout is reached.
while [ -z "$TARGET_POD" ]; do
    # **CORRECTION**: Use the correct label selector for a JobSet pod.
    # We target the first pod (job-index=0) of the 'worker' replicatedJob.
    POD_SELECTOR="jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/job-index=0"
    TARGET_POD=$(kubectl get pods -l ${POD_SELECTOR} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    
    if [ -z "$TARGET_POD" ]; then
        if [ $COUNTER -ge $TIMEOUT ]; then
            echo -e "${ORANGE}🛑 TIMEOUT: Worker pod was not found after ${TIMEOUT} seconds.${NC}"
            echo -e "${ORANGE}🔎 Final pod status check:${NC}"
            kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -o wide
            exit 1
        fi
        echo "Still waiting for worker pod... (${COUNTER}s / ${TIMEOUT}s)"
        sleep $INTERVAL
        COUNTER=$((COUNTER + INTERVAL))
    else
        echo -e "${GREEN}✅ Found target pod: ${TARGET_POD}${NC}"
    fi
done

echo -e "${ORANGE}📝 Waiting for pod '${TARGET_POD}' to be ready...${NC}"
kubectl wait --for=condition=Ready pod/${TARGET_POD} --timeout=300s

echo -e "${ORANGE}🪵 Tailing logs for pod '${TARGET_POD}'. Press Ctrl+C to stop.${NC}"
kubectl logs -f ${TARGET_POD}

echo -e "${GREEN}✅ Script finished.${NC}"
