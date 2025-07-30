#!/bin/bash
set -e

# --- Color definitions ---
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

## ------------------- Argument Parsing ------------------- ##
if [ "$#" -lt 1 ]; then
    echo -e "${ORANGE}Usage: $0 {shuffle | device-assignment} [--rebuild] [--cpu-eval]${NC}"
    exit 1
fi

CONFIG=$1
REBUILD_FLAG=false
CPU_EVAL_FLAG=false

for arg in "${@:2}"; do
    if [ "$arg" == "--rebuild" ]; then
        REBUILD_FLAG=true
    elif [ "$arg" == "--cpu-eval" ]; then
        CPU_EVAL_FLAG=true
    fi
done

## ------------------- Dynamic Configuration ------------------- ##
# Project and cluster details are hardcoded for this specific repro
export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_ZONE="us-east5-b"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export AR_REGION="us-east5"
export GCS_BUCKET_NAME="chavoshi-dlrm-training"
export GKE_LOCATION_FLAG="--zone ${CLUSTER_ZONE}"
export NAMESPACE="default"

# Shared configuration for the repro
export AR_REPO_NAME="tpu-repo"
export IMAGE_NAME="tpu-embedding-repro"
export IMAGE_TAG="latest"
export DOCKERFILE_NAME="Dockerfile"
export PYTHON_SCRIPT_NAME="training_code.py"
export YAML_FILE="tfjob-v6e-16-repro.yaml"

case $CONFIG in
    shuffle)
        export TFJOB_NAME="tpu-v6e-trainer-shuffle"
        export TRAINING_ENV_VARS="SHUFFLE_ENDPOINTS"
        ;;
    device-assignment)
        export TFJOB_NAME="tpu-v6e-trainer-da"
        export TRAINING_ENV_VARS="ENABLE_DEVICE_ASSIGNMENT"
        ;;
    *)
        echo -e "${ORANGE}Error: Invalid configuration '$CONFIG'. Choose 'shuffle' or 'device-assignment'.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}▶ Running with job configuration: $CONFIG${NC}"

## ------------------- Define Image & Checkpoint URLs ------------------- ##
export IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
export CHECKPOINT_DIR_GCS="gs://${GCS_BUCKET_NAME}/checkpoints-repro/${TFJOB_NAME}"

## ------------------- Authenticate & Configure ------------------- ##
echo -e "${BLUE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} ${GKE_LOCATION_FLAG} --project ${PROJECT_ID}
echo -e "${BLUE}🔐 Configuring Docker for ${AR_REGION}...${NC}"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

## ------------------- Conditional Docker Build & Push ------------------- ##
if [ "$REBUILD_FLAG" = true ] || ! docker image inspect "${IMAGE_URL}" &> /dev/null; then
    echo -e "${BLUE}🚀 Building Docker image from '${DOCKERFILE_NAME}'...${NC}"
    docker build -f "${DOCKERFILE_NAME}" -t "${IMAGE_URL}" . --platform linux/amd64
    echo -e "${BLUE}ଠ Pushing image to Artifact Registry...${NC}"
    docker push "${IMAGE_URL}"
    echo -e "${GREEN}✅ Image pushed to ${IMAGE_URL}.${NC}"
else
    echo -e "${GREEN}▶ Skipping Docker build. Using existing image: ${IMAGE_URL}${NC}"
fi

## ------------------- Generate Worker Hostnames ------------------- ##
WORKER_HOSTS=""
for i in {0..3}; do
    # --- NECESSARY CHANGE: Use the correct DNS suffix for a standard GKE cluster ---
    WORKER_HOSTS+="${TFJOB_NAME}-worker-${i}.${NAMESPACE}.svc,"
done
export TPU_WORKER_HOSTNAMES=${WORKER_HOSTS%?} # Remove trailing comma

## ------------------- Define Master Command ------------------- ##
export MASTER_COMMAND="python ${PYTHON_SCRIPT_NAME} --training --checkpoint-dir ${CHECKPOINT_DIR_GCS}"

## ------------------- TFJob Deployment & Logging ------------------- ##
echo -e "${BLUE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"
echo -e "${BLUE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} -n ${NAMESPACE} --ignore-not-found=true --wait=false
sleep 5

echo -e "${BLUE}🚢 Generating and deploying TFJob from template '${YAML_FILE}'...${NC}"
envsubst < "${YAML_FILE}" | kubectl apply -f -
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${BLUE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/job-name=${TFJOB_NAME},training.kubeflow.org/replica-type=master -n ${NAMESPACE} --timeout=15m

echo -e "${BLUE}🪵 Tailing logs for the MASTER pod...${NC}"
kubectl logs -f -l training.kubeflow.org/job-name=${TFJOB_NAME},training.kubeflow.org/replica-type=master -n ${NAMESPACE}

## ------------------- CPU Evaluation ------------------- ##
if [ "$CPU_EVAL_FLAG" = true ]; then
    echo -e "${BLUE}▶️  Training finished. Proceeding with CPU evaluation...${NC}"
    echo -e "${BLUE}⏳ Waiting for training job '${TFJOB_NAME}' to complete successfully...${NC}"

    kubectl wait --for=condition=Succeeded tfjob/${TFJOB_NAME} -n ${NAMESPACE} --timeout=30m

    echo -e "${GREEN}✅ Training job succeeded. Now running evaluation on a local CPU.${NC}"

    docker run --rm -it \
        -e USE_CPU_STRATEGY=true \
        -v ~/.config/gcloud:/root/.config/gcloud:ro \
        "${IMAGE_URL}" \
        python "${PYTHON_SCRIPT_NAME}" --checkpoint-dir "${CHECKPOINT_DIR_GCS}"
fi

echo -e "${GREEN}✅ All steps completed.${NC}"
