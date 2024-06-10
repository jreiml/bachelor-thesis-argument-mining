# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-1700
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6362158060073853,
        "f1_macro": 0.6756465772249152,
        "f1_true": 0.6513442456110593,
        "f1_false": 0.6999489088387709
    },
    "test_split_metric": {
        "threshold": 0.660316526889801,
        "f1_macro": 0.7081302101155912,
        "f1_true": 0.6865356161488058,
        "f1_false": 0.7297248040823766
    }
}
```

