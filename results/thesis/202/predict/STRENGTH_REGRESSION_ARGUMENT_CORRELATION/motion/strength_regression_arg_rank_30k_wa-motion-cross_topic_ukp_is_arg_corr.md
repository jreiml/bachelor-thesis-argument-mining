# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-1089
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6430716514587402,
        "f1_macro": 0.6953486693716843,
        "f1_true": 0.6694392042531299,
        "f1_false": 0.7212581344902387
    },
    "test_split_metric": {
        "threshold": 0.626078188419342,
        "f1_macro": 0.7180280015579198,
        "f1_true": 0.7008179078777443,
        "f1_false": 0.7352380952380952
    }
}
```

