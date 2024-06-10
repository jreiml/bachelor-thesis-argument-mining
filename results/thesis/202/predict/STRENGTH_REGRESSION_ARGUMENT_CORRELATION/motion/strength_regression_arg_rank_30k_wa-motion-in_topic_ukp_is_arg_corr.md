# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1564
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6386478543281555,
        "f1_macro": 0.6926316463952248,
        "f1_true": 0.671386430678466,
        "f1_false": 0.7138768621119835
    },
    "test_split_metric": {
        "threshold": 0.6623817086219788,
        "f1_macro": 0.7150173150115064,
        "f1_true": 0.7001239157372986,
        "f1_false": 0.7299107142857142
    }
}
```

