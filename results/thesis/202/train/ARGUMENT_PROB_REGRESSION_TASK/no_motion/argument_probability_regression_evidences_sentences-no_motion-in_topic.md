# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --task ARGUMENT_PROB_REGRESSION_TASK \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_probability_regression_evidences_sentences-no_motion-in_topic \
    --run_name argument_probability_regression_evidences_sentences-no_motion-in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_loss": 0.0390489399433136,
    "test_mean_squared_error": 0.0390489399433136,
    "test_root_mean_squared_error": 0.19760805368423462,
    "test_spearman": 0.7794927461380046,
    "test_pearson": 0.8035891023628994,
    "epochs_trained_total": 7.3782945736434105,
    "epochs_trained_best": 4.304005167958657
}
```

