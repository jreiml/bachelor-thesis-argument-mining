# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.21926647424697876,
        "f1_macro": 0.715047783769093,
        "f1_true": 0.6742365276315188,
        "f1_false": 0.755859039906667
    },
    "test_split_metric": {
        "threshold": 0.23798109591007233,
        "f1_macro": 0.7186085369037611,
        "f1_true": 0.6849010451412051,
        "f1_false": 0.7523160286663171
    }
}
```

