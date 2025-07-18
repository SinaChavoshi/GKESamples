
# Set Env variables
export TPU_NAME="chavoshi-dlrm-dnc-v2-benchmark"
export PROJECT="tpu-prod-env-one-vm"
export ZONE="us-east5-a"


# Pusher to update the remote workers
export REMOTE_DEST_PATH="~/RecML/recml/inference/models/jax/DLRM_DCNv2/" 
./sync_specific_files_scp.sh  dlrm_main.py 

export REMOTE_DEST_PATH="~/RecML/" 
./sync_specific_files_scp.sh  requirements.txt

export REMOTE_DEST_PATH="~/RecML/recml/inference/benchmarks/DLRM_DCNv2/"
./sync_specific_files_scp.sh   train_and_checkpoint.sh 


# Run the benchmark
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML/recml/inference/benchmarks/DLRM_DCNv2/ && chmod +x ./train_and_checkpoint.sh && cd ~/RecML && TPU_NAME=${TPU_NAME} ./recml/inference/benchmarks/DLRM_DCNv2/train_and_checkpoint.sh" 2>&1 | tee tpuv6e-16.txt

# exit to avoid killing the job
exit 0

# kill all jobs
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all   --command="pkill -f train_and_checkpoint.sh"
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all   --command="pkill -f dlrm_main.py"
