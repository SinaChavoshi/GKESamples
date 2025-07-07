export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export BUCKET_NAME="lora-finetuning-data-${PROJECT_ID}"
export K8S_NAMESPACE="default"
export JOB_NAME="lora-finetuning-job-jax-4x4"

# docker image build and push 
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/tpu-lora-trainer-jax:v4"
docker build . -t ${IMAGE_URI}
docker push ${IMAGE_URI}

# reload job 
kubectl delete job ${JOB_NAME}
kubectl apply -f tpu-training-job.yaml

kubectl get jobs
kubectl get pods
kubectl logs -f job/${JOB_NAME}

gcloud storage ls gs://${BUCKET_NAME}/output/final_checkpoint/


# Run data preprocessor locally
python preprocess_data.py \
    --input_path gs://${BUCKET_NAME}/raw_data/training_data.jsonl \
    --output_path gs://${BUCKET_NAME}/processed_data
