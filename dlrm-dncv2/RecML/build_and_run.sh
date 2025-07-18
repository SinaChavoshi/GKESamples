

export TPU_NAME="chavoshi-dlrm-dnc-v2-benchmark"
export PROJECT="tpu-prod-env-one-vm"
export ZONE="us-east5-a"

docker build -t us-east5-docker.pkg.dev/chavoshi-gke-dev/tpu-repo/jax-dlrm-gke:latest .
docker push us-east5-docker.pkg.dev/chavoshi-gke-dev/tpu-repo/jax-dlrm-gke:latest

kubectl delete jobset jax-dlrm-benchmark-v6e-32chip --ignore-not-found=true --wait=false

kubectl apply -f jobset.yaml

kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-dlrm-benchmark-v6e-32chip

sleep 10

kubectl logs -f jobs/jax-dlrm-benchmark-v6e-32chip-worker-0

# Pusher to update the remote workers
./sync_specific_files_scp.sh  dlrm_main.py 

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML/recml/inference/benchmarks/DLRM_DCNv2/ && chmod +x ./train_and_checkpoint.sh && cd ~/RecML && TPU_NAME=${TPU_NAME} ./recml/inference/benchmarks/DLRM_DCNv2/train_and_checkpoint.sh" 2>&1 | tee multihost.txt


# kill all jobs
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all   --command="pkill -f train_and_checkpoint.sh"
