# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-1782
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.1845393180847168,
        "f1_macro": 0.7062851044775269,
        "f1_true": 0.6709476031215162,
        "f1_false": 0.7416226058335377
    },
    "test_split_metric": {
        "threshold": 0.2311529964208603,
        "f1_macro": 0.7166040493060092,
        "f1_true": 0.6752234700893881,
        "f1_false": 0.7579846285226303
    }
}
```

