# Set environment variables from your setup
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export BUCKET_NAME="dlrm-training-${PROJECT_ID}"
export IMAGE_NAME="jax-dlrm-gke"

docker build -f Dockerfile -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:latest

envsubst < jobset_jax.yaml | kubectl apply -f -
envsubst < jobset_jax_singlenode_test.yaml | kubectl apply -f -

# Check the status of the JobSet
kubectl get jobset jax-dlrm-benchmark

kubectl get pods -w 

# Check the logs of one of the pods
kubectl logs -f jobset/jax-dlrm-benchmark

kubectl delete jobset jax-dlrm-benchmark --wait=false
kubectl delete jobsets.jobset.x-k8s.io jax-dlrm-singlenode-test --wait=false

# Generate and save synthatic data 
# 1. Run the Python script to create the local .tsv files
python3 generate_synthetic_data.py


# 3. Upload the generated files to a new directory in your GCS bucket
gsutil cp train_sample.tsv gs://${BUCKET_NAME}/synthetic_data/train/
gsutil cp eval_sample.tsv gs://${BUCKET_NAME}/synthetic_data/eval/

echo "Synthetic data successfully uploaded to GCS."
