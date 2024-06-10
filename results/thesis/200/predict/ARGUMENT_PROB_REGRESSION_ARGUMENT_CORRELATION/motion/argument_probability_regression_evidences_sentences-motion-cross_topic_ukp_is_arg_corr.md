# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-1188
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.23848363757133484,
        "f1_macro": 0.724838717378588,
        "f1_true": 0.6919169414490687,
        "f1_false": 0.7577604933081072
    },
    "test_split_metric": {
        "threshold": 0.25692835450172424,
        "f1_macro": 0.7554269148923747,
        "f1_true": 0.735358872743285,
        "f1_false": 0.7754949570414643
    }
}
```

