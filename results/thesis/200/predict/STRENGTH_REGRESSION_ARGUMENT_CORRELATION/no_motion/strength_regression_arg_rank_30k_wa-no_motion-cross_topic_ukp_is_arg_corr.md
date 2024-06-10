# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6757199764251709,
        "f1_macro": 0.6894908877351563,
        "f1_true": 0.6538137658999247,
        "f1_false": 0.725168009570388
    },
    "test_split_metric": {
        "threshold": 0.6435333490371704,
        "f1_macro": 0.7227525529912195,
        "f1_true": 0.7159726538222497,
        "f1_false": 0.7295324521601894
    }
}
```

