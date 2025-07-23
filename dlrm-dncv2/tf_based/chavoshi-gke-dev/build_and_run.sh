#!/bin/bash
set -e
# --- Configuration ---
export PROJECT_ID="tpu-vm-gke-testing"
export CLUSTER_REGION="us-central2"
export CLUSTER_NAME="chavoshi-test"
export AR_REGION="us-central2"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export BUCKET="chavoshi-checkpoints"
export MASTER_IMAGE_NAME="dlrm-master"
export WORKER_IMAGE_NAME="dlrm-worker"
export YAML_FILE="tfjob_checkpoint_repro.yaml"
export TFJOB_NAME="tf-final-dlrm-job"

# --- Connect and Auth ---
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${CLUSTER_REGION} --project ${PROJECT_ID}
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

# --- Build and Push ---
MASTER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${MASTER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f master.Dockerfile -t ${MASTER_IMAGE_URL} .

WORKER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${WORKER_IMAGE_NAME}:${IMAGE_TAG}"
docker build -f worker.Dockerfile -t ${WORKER_IMAGE_URL} .
docker push ${WORKER_IMAGE_URL}

# --- Deploy and Log ---
kubectl delete tfjob ${TFJOB_NAME} -n default --ignore-not-found=true && sleep 5
kubectl apply -f ${YAML_FILE}
echo "Waiting for Master pod..."
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=15m
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}
