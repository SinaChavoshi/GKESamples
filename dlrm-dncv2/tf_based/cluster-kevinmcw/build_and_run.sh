#!/bin/bash
set -e

## ------------------- Configuration ------------------- ##
export PROJECT_ID="tpu-vm-gke-testing"
export ZONE="us-central2-b"
export CLUSTER_NAME="cluster-kevinmcw"
export YAML_FILE="tfjob_checkpoint_repro.yaml"
export TFJOB_NAME="tf-16-kevinmcw-tfjob"

# Color definitions
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m'

## ----------- Connect and Deploy ----------- ##
echo -e "${ORANGE}🔌 Connecting to colleague's GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

echo -e "${ORANGE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"
echo -e "${ORANGE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} -n default --ignore-not-found=true
sleep 5

echo -e "${ORANGE}🚢 Deploying TFJob from '${YAML_FILE}'...${NC}"
kubectl apply -f ${YAML_FILE}
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=15m

echo -e "${ORANGE}🪵 Tailing logs for the MASTER pod. Press Ctrl+C to stop.${NC}"
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}

echo -e "${GREEN}✅ Script finished.${NC}"
