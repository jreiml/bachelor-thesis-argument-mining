# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-1632
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.70982426404953,
        "f1_macro": 0.666807094291042,
        "f1_true": 0.6405484818805093,
        "f1_false": 0.6930657067015746
    },
    "test_split_metric": {
        "threshold": 0.7440568804740906,
        "f1_macro": 0.6978256130385183,
        "f1_true": 0.6672473867595818,
        "f1_false": 0.7284038393174547
    }
}
```

