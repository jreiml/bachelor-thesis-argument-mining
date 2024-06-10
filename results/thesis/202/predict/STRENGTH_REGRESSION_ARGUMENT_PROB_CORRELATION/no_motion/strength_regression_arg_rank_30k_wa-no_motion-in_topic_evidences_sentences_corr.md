# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-1700
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.24045896530151367,
        "root_mean_squared_error": 0.4903661608695984,
        "spearman": 0.3436538194649504,
        "pearson": 0.31100843354250246
    },
    "test_split_metric": {
        "mean_squared_error": 0.2450712025165558,
        "root_mean_squared_error": 0.4950466752052307,
        "spearman": 0.33402202792390717,
        "pearson": 0.2932148410601012
    }
}
```

