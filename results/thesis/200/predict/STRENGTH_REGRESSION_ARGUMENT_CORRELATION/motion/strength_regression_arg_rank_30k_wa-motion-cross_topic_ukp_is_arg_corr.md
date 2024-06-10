# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6432104110717773,
        "f1_macro": 0.7011873166053851,
        "f1_true": 0.665717083592985,
        "f1_false": 0.7366575496177853
    },
    "test_split_metric": {
        "threshold": 0.6264173984527588,
        "f1_macro": 0.7178460006010667,
        "f1_true": 0.7007958700795871,
        "f1_false": 0.7348961311225463
    }
}
```

