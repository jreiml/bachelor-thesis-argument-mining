# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-1188
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.12055521458387375,
        "root_mean_squared_error": 0.3472106158733368,
        "spearman": 0.2790907740534869,
        "pearson": 0.291824370924485
    },
    "test_split_metric": {
        "mean_squared_error": 0.13647310435771942,
        "root_mean_squared_error": 0.36942267417907715,
        "spearman": 0.28860069487358087,
        "pearson": 0.30368056232737245
    }
}
```

