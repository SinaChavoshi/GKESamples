
export PROJECT_ID="chavoshi-gke-dev"
export PROJECT="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export BUCKET_NAME="dlrm-training-${PROJECT_ID}"
export IMAGE_NAME="jax-dlrm-gke"
export ZONE="us-east5-a"
# export TPU_NAME="v6e16"
export TPU_NAME="chavoshi-dlrm-dnc-v2-benchmark"

docker build -t us-east5-docker.pkg.dev/chavoshi-gke-dev/tpu-repo/jax-dlrm-gke:latest .
docker push us-east5-docker.pkg.dev/chavoshi-gke-dev/tpu-repo/jax-dlrm-gke:latest

kubectl delete jobset jax-dlrm-benchmark-v6e-32chip --ignore-not-found=true --wait=false

kubectl apply -f jobset.yaml

kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-dlrm-benchmark-v6e-32chip

sleep 10

kubectl logs -f jobs/jax-dlrm-benchmark-v6e-32chip-worker-0

# Pusher to update the remote workers
./sync_specific_files_scp.sh  dlrm_main.py 
