# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-1632
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.3253418803215027,
        "root_mean_squared_error": 0.5703874826431274,
        "spearman": 0.36014341364298497,
        "pearson": 0.3172446103425423
    },
    "test_split_metric": {
        "mean_squared_error": 0.32906925678253174,
        "root_mean_squared_error": 0.5736455917358398,
        "spearman": 0.3541674485842664,
        "pearson": 0.30848014210929164
    }
}
```

