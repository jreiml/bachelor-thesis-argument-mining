# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --task STRENGTH_REGRESSION_TASK \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/strength_regression_arg_rank_30k_wa-motion-in_topic \
    --run_name strength_regression_arg_rank_30k_wa-motion-in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 20 \
    --max_len_percentile 99 \
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_loss": 0.024968622252345085,
    "test_mean_squared_error": 0.024968622252345085,
    "test_root_mean_squared_error": 0.15801462531089783,
    "test_spearman": 0.5148454360112887,
    "test_pearson": 0.5626080389025799,
    "epochs_trained_total": 7.430493273542601,
    "epochs_trained_best": 5.394741691750108
}
```

