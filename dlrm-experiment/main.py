# main.py
import jax
import fiddle as fdl
import numpy as np
import recml
import dlrm_experiment

def main():
    """Initializes the distributed environment and runs the DLRM experiment."""
    print("Initializing JAX for distributed training...")
    jax.distributed.initialize()

    print(f"JAX initialized. Process {jax.process_index()} of {jax.process_count()}.")
    print(f"Global device count: {jax.device_count()}, Local device count: {jax.local_device_count()}")

    # Seed for reproducibility
    np.random.seed(1337 + jax.process_index())

    experiment_config = dlrm_experiment.experiment()

    # To reduce table sizes for a quick test:
    # for cfg in fdl.selectors.select(experiment_config, dlrm_experiment.SparseFeature):
    #   cfg.vocab_size = 2000
    #   cfg.embedding_dim = 16
    
    print("Building the experiment from configuration...")
    experiment = fdl.build(experiment_config)

    print("Starting experiment run...")
    recml.run_experiment(experiment, mode=recml.Experiment.Mode.TRAIN_AND_EVAL)
    
    print("Experiment finished.")

if __name__ == "__main__":
    main()
