#!/bin/bash
set -e

# --- Color definitions ---
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

## ------------------- Argument Parsing ------------------- ##
if [ "$#" -lt 2 ]; then
    echo -e "${ORANGE}Usage: $0 {test} {shuffle | device-assignment} [--rebuild] [--cpu-eval]${NC}"
    echo -e "${ORANGE}  - 'shuffle': Run the failing scenario using endpoint shuffling.${NC}"
    echo -e "${ORANGE}  - 'device-assignment': Run the working scenario using DeviceAssignment.${NC}"
    echo -e "${ORANGE}  - '--rebuild': Force a rebuild of the Docker image.${NC}"
    echo -e "${ORANGE}  - '--cpu-eval': Run evaluation on CPU after training completes.${NC}"
    exit 1
fi

PROJECT_ALIAS=$1
CONFIG=$2
REBUILD_FLAG=false
CPU_EVAL_FLAG=false

# Check for flags starting from the 3rd argument
for arg in "${@:3}"; do
    if [ "$arg" == "--rebuild" ]; then
        REBUILD_FLAG=true
    elif [ "$arg" == "--cpu-eval" ]; then
        CPU_EVAL_FLAG=true
    fi
done

## ------------------- Dynamic Configuration ------------------- ##
echo -e "${GREEN}▶ Setting up configuration for project alias: '${PROJECT_ALIAS}'${NC}"

# Set project-specific variables based on the chosen alias
case $PROJECT_ALIAS in
    test)
        export PROJECT_ID="tpu-vm-gke-testing"
        export CLUSTER_REGION="us-central2"
        export CLUSTER_NAME="chavoshi-test"
        export AR_REGION="us-central1"
        export GCS_BUCKET_NAME="kkukreja-playground-us-central2"
        export GKE_LOCATION_FLAG="--region ${CLUSTER_REGION}"
        ;;
    *)
        echo -e "${ORANGE}Error: Invalid project alias '$PROJECT_ALIAS'. Choose 'test'.${NC}"
        exit 1
        ;;
esac

# Shared configuration
export AR_REPO_NAME="tpu-repro-repo"
export IMAGE_NAME="tpu-embedding-repro"
export IMAGE_TAG="latest"
export DOCKERFILE_NAME="Dockerfile"
export PYTHON_SCRIPT_NAME="repro_script.py"
export YAML_FILE="tfjob-v6e-16-repro.yaml"

# Set Job Name and Training Params based on the selected configuration
case $CONFIG in
    shuffle)
        export TFJOB_NAME="tpu-repro-shuffle"
        export TRAINING_ENV_VARS="SHUFFLE_ENDPOINTS=true"
        ;;
    device-assignment)
        export TFJOB_NAME="tpu-repro-da"
        export TRAINING_ENV_VARS="ENABLE_DEVICE_ASSIGNMENT=true"
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

## ------------------- Ensure Artifact Registry Repo Exists ------------------- ##
echo -e "${BLUE}🔎 Checking for Artifact Registry repository '${AR_REPO_NAME}'...${NC}"
if ! gcloud artifacts repositories describe ${AR_REPO_NAME} --location=${AR_REGION} --project=${PROJECT_ID} &> /dev/null
then
    echo -e "${ORANGE}Repository not found. Creating '${AR_REPO_NAME}'...${NC}"
    gcloud artifacts repositories create ${AR_REPO_NAME} \
        --repository-format=docker \
        --location=${AR_REGION} \
        --project=${PROJECT_ID} \
        --description="Docker repository for TPU embedding repro"
    echo -e "${GREEN}✅ Repository created successfully.${NC}"
else
    echo -e "${GREEN}✅ Repository already exists.${NC}"
fi

## ------------------- Conditional Docker Build & Push ------------------- ##
if [ "$REBUILD_FLAG" = true ] || ! docker image inspect "${IMAGE_URL}" &> /dev/null; then
    echo -e "${BLUE}🚀 Building Docker image...${NC}"
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
    WORKER_HOSTS+="${TFJOB_NAME}-worker-${i}.training.svc,"
done
export TPU_WORKER_HOSTNAMES=${WORKER_HOSTS%?} # Remove trailing comma

## ------------------- Define Training Command ------------------- ##
export TRAINING_COMMAND="${TRAINING_ENV_VARS} python ${PYTHON_SCRIPT_NAME} --training --checkpoint-dir ${CHECKPOINT_DIR_GCS}"

## ------------------- TFJob Deployment & Logging ------------------- ##
echo -e "${BLUE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"
echo -e "${BLUE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} -n default --ignore-not-found=true --wait=false
sleep 5

echo -e "${BLUE}🚢 Generating and deploying TFJob from template '${YAML_FILE}'...${NC}"
envsubst < "${YAML_FILE}" | kubectl apply -f -
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${BLUE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=15m

echo -e "${BLUE}🪵 Tailing logs for the MASTER pod. Training output appears below. Press Ctrl+C to stop viewing logs.${NC}"
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}

## ------------------- CPU Evaluation ------------------- ##
if [ "$CPU_EVAL_FLAG" = true ]; then
    echo -e "${BLUE}▶️  Training finished. Proceeding with CPU evaluation as requested.${NC}"
    echo -e "${BLUE}⏳ Waiting for training job '${TFJOB_NAME}' to complete successfully...${NC}"

    kubectl wait --for=condition=Succeeded tfjob/${TFJOB_NAME} --timeout=30m
    
    echo -e "${GREEN}✅ Training job succeeded. Now running evaluation on a local CPU.${NC}"
    echo -e "${ORANGE}Note: Make sure gcloud is authenticated for GCS access ('gcloud auth application-default login').${NC}"

    docker run --rm -it \
        -e USE_CPU_STRATEGY=true \
        -v ~/.config/gcloud:/root/.config/gcloud:ro \
        "${IMAGE_URL}" \
        python "${PYTHON_SCRIPT_NAME}" --checkpoint-dir "${CHECKPOINT_DIR_GCS}"
else
    echo -e "${GREEN}▶ Script finished. To run CPU evaluation, re-run with the --cpu-eval flag.${NC}"
fi

echo -e "${GREEN}✅ All steps completed.${NC}"
