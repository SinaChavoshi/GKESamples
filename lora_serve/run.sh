export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export BUCKET_NAME="lora-finetuning-data-${PROJECT_ID}"
export K8S_NAMESPACE="default"
export JOB_NAME="lora-finetuning-job"

export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/lora-chat:latest"

docker build . -t $IMAGE_URI
docker push $IMAGE_URI

kubectl apply -f deployment.yaml

POD_NAME=$(kubectl get pods -l app=lora-chat -o jsonpath='{.items[0].metadata.name}')


kubectl attach -it $POD_NAME
kubectl exec -it $POD_NAME -- /bin/bash

# kubectl delete pod $POD_NAME
