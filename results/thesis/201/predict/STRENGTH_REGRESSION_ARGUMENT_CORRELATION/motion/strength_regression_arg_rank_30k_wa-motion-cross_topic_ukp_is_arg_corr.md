# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-660
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.7153100371360779,
        "f1_macro": 0.6762140819274225,
        "f1_true": 0.6400529918304261,
        "f1_false": 0.7123751720244188
    },
    "test_split_metric": {
        "threshold": 0.7120411396026611,
        "f1_macro": 0.7023008171961297,
        "f1_true": 0.680340388391883,
        "f1_false": 0.7242612460003763
    }
}
```

