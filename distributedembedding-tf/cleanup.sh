# Clean up the old components
kubectl delete jobset multislice-job --ignore-not-found=true
kubectl delete configmap jax-script-config --ignore-not-found=true
kubectl delete service multislice-job-svc --ignore-not-found=true
