export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export BUCKET_NAME="lora-finetuning-data-${PROJECT_ID}"
export K8S_NAMESPACE="default"

# docker image build and push 
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/tpu-lora-trainer:v7"
docker build . -t ${IMAGE_URI}
docker push ${IMAGE_URI}

# reload job 
kubectl delete job lora-finetuning-job
kubectl apply -f tpu-training-job.yaml

kubectl get jobs
kubectl get pods
kubectl logs -f job/lora-finetuning-job

gcloud storage ls gs://${BUCKET_NAME}/output/final_checkpoint/
