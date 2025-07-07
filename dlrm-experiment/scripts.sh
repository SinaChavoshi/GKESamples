export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export BUCKET_NAME="dlrm-training-${PROJECT_ID}"
export K8S_NAMESPACE="default"
export JOB_NAME="dlrm-jax"

# docker image build and push 
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/dlrm-jax:latest"
docker build . -t ${IMAGE_URI}
docker push ${IMAGE_URI}

# reload job 
kubectl delete job ${JOB_NAME}
kubectl apply -f jobset.yaml

kubectl get jobs
kubectl get pods
kubectl logs -f job/${JOB_NAME}

