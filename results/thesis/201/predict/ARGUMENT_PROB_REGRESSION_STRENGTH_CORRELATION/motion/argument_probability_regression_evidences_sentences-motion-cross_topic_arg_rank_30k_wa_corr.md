# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-858
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.1160060465335846,
        "root_mean_squared_error": 0.34059661626815796,
        "spearman": 0.28127487287274616,
        "pearson": 0.29113776486650583
    },
    "test_split_metric": {
        "mean_squared_error": 0.1290743499994278,
        "root_mean_squared_error": 0.3592692017555237,
        "spearman": 0.27448672663008966,
        "pearson": 0.2870852548719203
    }
}
```

