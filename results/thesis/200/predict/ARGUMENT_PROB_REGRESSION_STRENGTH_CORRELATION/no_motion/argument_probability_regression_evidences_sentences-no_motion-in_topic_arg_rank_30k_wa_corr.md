# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-1782
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.14228521287441254,
        "root_mean_squared_error": 0.3772071301937103,
        "spearman": 0.23519989236445835,
        "pearson": 0.24204177631333562
    },
    "test_split_metric": {
        "mean_squared_error": 0.1455714851617813,
        "root_mean_squared_error": 0.3815383017063141,
        "spearman": 0.22656105329513773,
        "pearson": 0.23379201941875097
    }
}
```

