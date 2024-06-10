# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.12497904151678085,
        "root_mean_squared_error": 0.35352376103401184,
        "spearman": 0.3148818664241083,
        "pearson": 0.32372342295900736
    },
    "test_split_metric": {
        "mean_squared_error": 0.13000063598155975,
        "root_mean_squared_error": 0.3605560064315796,
        "spearman": 0.3013105983497288,
        "pearson": 0.31020638800371036
    }
}
```

