"""Main script to run the DLRM experiment."""

from absl import app
from absl import flags
import jax
import fiddle as fdl
import numpy as np
import recml
import dlrm_experiment

# Define command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None, 'Directory to save checkpoints.', required=True)
flags.DEFINE_string('train_data_path', None, 'Path to the training data.', required=True)
flags.DEFINE_string('eval_data_path', None, 'Path to the evaluation data.', required=True)
flags.DEFINE_integer('global_batch_size', 16384, 'Global batch size for training.')
flags.DEFINE_integer('train_steps', 10000, 'Total number of training steps.')
flags.DEFINE_integer('steps_per_eval', 1000, 'Number of steps between evaluations.')

def main(_):
    """Initializes JAX, configures, and runs the DLRM experiment."""
    print("--- Initializing JAX for distributed training ---")
    jax.distributed.initialize()
    print(f"JAX initialized. Process {jax.process_index()}/{jax.process_count()}.")
    print(f"Global devices: {jax.device_count()}, Local devices: {jax.local_device_count()}")

    # Seed for reproducibility, ensuring each process has a different seed
    np.random.seed(42 + jax.process_index())

    # Get the base experiment configuration
    experiment_config = dlrm_experiment.experiment()

    # Use Fiddle to patch the configuration with values from flags
    # This is a clean way to override defaults without modifying the main experiment file
    experiment_config.task.train_data.global_batch_size = FLAGS.global_batch_size
    experiment_config.task.eval_data.global_batch_size = FLAGS.global_batch_size
    experiment_config.task.train_data.input_path = FLAGS.train_data_path
    experiment_config.task.eval_data.input_path = FLAGS.eval_data_path
    experiment_config.trainer.checkpointer.logdir = FLAGS.model_dir
    experiment_config.trainer.train_steps = FLAGS.train_steps
    experiment_config.trainer.steps_per_eval = FLAGS.steps_per_eval

    print("--- Final Experiment Configuration ---")
    print(fdl.repr(experiment_config))
    
    print("Building the experiment from configuration...")
    experiment = fdl.build(experiment_config)

    print("Starting experiment run...")
    recml.run_experiment(experiment, mode=recml.Experiment.Mode.TRAIN_AND_EVAL)
    
    print("✅ Experiment finished successfully.")

if __name__ == "__main__":
    app.run(main)
