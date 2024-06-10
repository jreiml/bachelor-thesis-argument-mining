#!/bin/bash

python3 src/models/train_model.py --input_dataset_csv "${2}" \
  --task STRENGTH_REGRESSION_TASK \
  --model_name bert-large-uncased \
  --include_motion "${1:-False}" \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 64 \
  --evaluation_strategy steps \
  --logging_epoch_step 0.1 \
  --learning_rate 1e-05 \
  --num_train_epochs 10 \
  --output_dir "models/strength_regression_arg_rank_30k_wa${3}" \
  --run_name "strength_regression_arg_rank_30k_wa${3}" \
  --load_best_model_at_end 1 \
  --save_total_limit 1 \
  --mlflow_env .mlflow \
  --early_stopping_patience 20 \
  --max_len_percentile 99 \
  --seed ${4} \
  --result_prefix ${4}
