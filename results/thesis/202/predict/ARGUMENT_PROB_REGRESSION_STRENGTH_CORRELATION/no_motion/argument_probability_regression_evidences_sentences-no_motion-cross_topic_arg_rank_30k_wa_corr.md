# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.13300201296806335,
        "root_mean_squared_error": 0.36469441652297974,
        "spearman": 0.22291854311371428,
        "pearson": 0.22793287821880923
    },
    "test_split_metric": {
        "mean_squared_error": 0.1556643545627594,
        "root_mean_squared_error": 0.39454323053359985,
        "spearman": 0.19585024168113693,
        "pearson": 0.19865513227942622
    }
}
```

