# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1802
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.703353762626648,
        "f1_macro": 0.6780384985317397,
        "f1_true": 0.6377617360893153,
        "f1_false": 0.7183152609741641
    },
    "test_split_metric": {
        "threshold": 0.7219150066375732,
        "f1_macro": 0.703346757028531,
        "f1_true": 0.6704820603125687,
        "f1_false": 0.7362114537444934
    }
}
```

