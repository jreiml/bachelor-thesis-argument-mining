# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-924
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.751282811164856,
        "f1_macro": 0.6609628798598219,
        "f1_true": 0.6290168904287571,
        "f1_false": 0.6929088692908869
    },
    "test_split_metric": {
        "threshold": 0.7353434562683105,
        "f1_macro": 0.690867453687785,
        "f1_true": 0.6848072562358277,
        "f1_false": 0.6969276511397424
    }
}
```

