#!/bin/bash

PROJECT_ROOT="/home/ubuntu/verl"
MODEL_PATH="/home/ubuntu/verl/model_cache"
DATA_PATH="/home/ubuntu/data"

# export PYTHONPATH="$PROJECT_ROOT/aero_framework/PGS/verl:$PROJECT_ROOT/aero_framework/PGS"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="$MODEL_PATH/Qwen2.5-1.5B"
export CUDA_VISIBLE_DEVICES=0,1
export MATH_VERIFY_TIMEOUT=300
# export OMP_NUM_THREADS=4


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export WANDB_DIR="$PROJECT_ROOT/aero_framework/PGS/wandb/run-20250907_155109-j6xp9n82"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="GRPO-256-16g-openr1"
LOG_DIR="/home/ubuntu/projects/verl/logs"
LOG_FILE="$LOG_DIR/grpo_baseline_${TIMESTAMP}_${EXPERIMENT_NAME}.log"
mkdir -p "$LOG_DIR"

echo "Starting GRPO Baseline test..."
echo "Algorithm: Standard GRPO (PGS infrastructure)"
echo "Log file: $LOG_FILE"

cd "$PROJECT_ROOT/aero_framework/PGS"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/openr1.parquet \
    data.val_files=$DATA_PATH/openr1_test1.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=150 \
    data.max_prompt_length=1024 \
    data.max_response_length=1536 \
    data.shuffle=True \
    +data.val_shuffle=True \
    +data.resampling_func=1 \
    +data.use_template=True \
    +data.reward_impl_version=2 \
    +actor_rollout_ref.ref.use_ref=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.ref.fsdp_config.fsdp_size=2 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +algorithm.n_max=32 \
    +algorithm.grpo_use_std=False \
    +algorithm.clip_adv_value=1.0 \
    trainer.critic_warmup=0 \
    +trainer.del_last_ckpt=False \
    +trainer.log_train=True \
    trainer.rejection_sample=False \
    trainer.logger=[console,wandb] \
    trainer.project_name='TEST' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.val_before_train=False \
    +trainer.disable_actor_update=False \
    +actor_rollout_ref.actor.skip_nan_update=False \
    +actor_rollout_ref.actor.verbose=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=90 \
    +trainer.init_global_steps=0 \
    trainer.total_epochs=10 "${@:1}" > "$LOG_FILE" 2>&1 &

PID=$!
echo "GRPO Baseline training started with PID: $PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop training: kill $PID"
echo $PID > "$LOG_DIR/grpo_baseline.pid"

echo ""
echo "=== GRPO BASELINE CONFIGURATION ==="
echo "✅ Algorithm: Standard GRPO (no rejection sampling)"
echo "✅ Same model checkpoint as rejection experiment"
echo "✅ Same data, batch size, and training parameters"
echo "✅ Direct comparison with rejection sampling"
echo "✅ Project: PGS_TEST"
echo "✅ Experiment: grpo-baseline"
echo ""
echo "=== COMPARISON SETUP ==="
echo "Both experiments start from the same checkpoint (global_step_200)"
echo "- Rejection: Uses rejection sampling + rebalancing"
echo "- Baseline: Uses standard GRPO without modification"
echo "- Same training data, batch size, learning rate"
echo "- Same validation frequency and metrics"
echo "- Results will be comparable in WandB"
Collapse












