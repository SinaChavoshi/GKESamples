
export IMAGE_NAME=us-east5-docker.pkg.dev/chavoshi-gke-dev/tpu-repo/embedding-poc:latest
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME
kubectl delete jobset multislice-job
kubectl apply -f tpu-multislice.yaml
kubectl get pods -w
