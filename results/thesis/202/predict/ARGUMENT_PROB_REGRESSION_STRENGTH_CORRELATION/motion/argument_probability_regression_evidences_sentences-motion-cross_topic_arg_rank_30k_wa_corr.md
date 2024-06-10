# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-924
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.11504527181386948,
        "root_mean_squared_error": 0.3391832411289215,
        "spearman": 0.2902528029205235,
        "pearson": 0.3012636037621129
    },
    "test_split_metric": {
        "mean_squared_error": 0.12334217131137848,
        "root_mean_squared_error": 0.35120102763175964,
        "spearman": 0.29665359761857657,
        "pearson": 0.3138946167380245
    }
}
```

