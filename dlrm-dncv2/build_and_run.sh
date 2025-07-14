# Set environment variables from your setup
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export AR_REPO_NAME="tpu-repo"
export IMAGE_NAME="tf-dlrm-gke"
export IMAGE_TAG="latest"
export BUCKET_NAME="dlrm-training-${PROJECT_ID}"

# Build the Docker image
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG} .

# Push the image to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

envsubst < jobset.yaml | kubectl apply -f -
