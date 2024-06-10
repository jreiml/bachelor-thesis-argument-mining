#!/bin/bash

python3 src/models/train_model.py --input_dataset_csv "${2}" "${3}" "${4}" \
  --input_dataset_task_head 0 1 2 \
  --task ALL_MULTI_TASK \
  --task_head_loss_weights ${5} ${6} ${7} \
  --model_name bert-large-uncased \
  --include_motion "${1:-False}" \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy steps \
  --logging_epoch_step 0.1 \
  --learning_rate 1e-05 \
  --num_train_epochs 10 \
  --output_dir "models/multitask${8}" \
  --run_name "multitask${8}" \
  --load_best_model_at_end 1 \
  --save_total_limit 1 \
  --mlflow_env .mlflow \
  --early_stopping_patience 30 \
  --max_len_percentile 99 \
  --seed ${9} \
  --result_prefix ${9}

