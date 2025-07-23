#!/bin/bash
export PYTHONPATH=/recommenders/:/models/
export TF_XLA_FLAGS='--tf_mlir_enable_mlir_bridge=true --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true'

TF_USE_LEGACY_KERAS=1 TPU_LOAD_LIBRARY=0 python3 ./models/official/recommendation/ranking/train.py  --mode=train     --model_dir=/tmp --params_override="
runtime:
  distribution_strategy: tpu
  mixed_precision_dtype: 'mixed_bfloat16'
  tpu: 'grpc://tf-16-dlrm-tfjob-worker-0.default.svc,grpc://tf-16-dlrm-tfjob-worker-1.default.svc,grpc://tf-16-dlrm-tfjob-worker-2.default.svc,grpc://tf-16-dlrm-tfjob-worker-3.default.svc' # This will be adapted by the TFJob name
task:
  use_synthetic_data: true # Using synthetic data to avoid GCS dependency for this test
  model:
    num_dense_features: 13
    bottom_mlp: [512, 256, 128]
    embedding_dim: 128
    interaction: 'dot'
    top_mlp: [1024, 1024, 512, 256, 1]
    vocab_sizes: [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1542, 63, 38532951, 2953546, 403346, 10, 2208, 11938, 154, 4, 975, 14, 39979771, 39956499, 39972858, 582458, 12969, 108, 36]
trainer:
  train_steps: 1000
  validation_interval: 200
  validation_steps: 100
  steps_per_loop: 200
"
