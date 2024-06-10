# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-924
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.2727792263031006,
        "f1_macro": 0.7125502725051682,
        "f1_true": 0.6651005450103364,
        "f1_false": 0.7599999999999999
    },
    "test_split_metric": {
        "threshold": 0.20367088913917542,
        "f1_macro": 0.7576092369786741,
        "f1_true": 0.746880947346162,
        "f1_false": 0.7683375266111863
    }
}
```

