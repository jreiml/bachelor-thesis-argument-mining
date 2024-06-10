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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_loss": 0.02573559619486332,
    "test_mean_squared_error": 0.02573559619486332,
    "test_root_mean_squared_error": 0.1604231745004654,
    "test_spearman": 0.49616229019541713,
    "test_pearson": 0.5435008691778487,
    "epochs_trained_total": 4.681614349775785,
    "epochs_trained_best": 2.6461298498732697
}
```

