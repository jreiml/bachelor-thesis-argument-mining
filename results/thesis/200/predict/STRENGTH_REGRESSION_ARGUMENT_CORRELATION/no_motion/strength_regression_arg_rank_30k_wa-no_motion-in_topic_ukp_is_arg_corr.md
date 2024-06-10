# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-884
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.680692732334137,
        "f1_macro": 0.6884827572095462,
        "f1_true": 0.6529708669087044,
        "f1_false": 0.723994647510388
    },
    "test_split_metric": {
        "threshold": 0.660210132598877,
        "f1_macro": 0.7166282364688554,
        "f1_true": 0.7101942719807731,
        "f1_false": 0.7230622009569377
    }
}
```

