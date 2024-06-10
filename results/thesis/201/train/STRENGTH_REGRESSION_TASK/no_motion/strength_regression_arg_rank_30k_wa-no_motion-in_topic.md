# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
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
    --output_dir models/strength_regression_arg_rank_30k_wa-no_motion-in_topic \
    --run_name strength_regression_arg_rank_30k_wa-no_motion-in_topic \
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
    "test_loss": 0.025527644902467728,
    "test_mean_squared_error": 0.025527644902467728,
    "test_root_mean_squared_error": 0.15977373719215393,
    "test_spearman": 0.5034469398536207,
    "test_pearson": 0.5447642118910021,
    "epochs_trained_total": 6.920777279521674,
    "epochs_trained_best": 4.885254550250594
}
```

