# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-1089
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.696328341960907,
        "f1_macro": 0.6760832308993385,
        "f1_true": 0.6344691002385561,
        "f1_false": 0.7176973615601209
    },
    "test_split_metric": {
        "threshold": 0.6889222860336304,
        "f1_macro": 0.6935096343678433,
        "f1_true": 0.667989417989418,
        "f1_false": 0.7190298507462687
    }
}
```

