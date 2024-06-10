# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.14035870134830475,
        "root_mean_squared_error": 0.37464475631713867,
        "spearman": 0.2432675090250816,
        "pearson": 0.2520052430448162
    },
    "test_split_metric": {
        "mean_squared_error": 0.1446613222360611,
        "root_mean_squared_error": 0.3803436756134033,
        "spearman": 0.23295437402463845,
        "pearson": 0.23900923554629366
    }
}
```

