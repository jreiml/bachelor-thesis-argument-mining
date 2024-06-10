# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --task STRENGTH_REGRESSION_TASK \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic \
    --run_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 20 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_loss": 0.028771281242370605,
    "test_mean_squared_error": 0.028771281242370605,
    "test_root_mean_squared_error": 0.16962099075317383,
    "test_spearman": 0.47688658835986597,
    "test_pearson": 0.5300484920459488,
    "epochs_trained_total": 5.3323170731707314,
    "epochs_trained_best": 3.3201219512195124
}
```

