# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.19416840374469757,
        "f1_macro": 0.7019969347782664,
        "f1_true": 0.6621688929137424,
        "f1_false": 0.7418249766427903
    },
    "test_split_metric": {
        "threshold": 0.20165246725082397,
        "f1_macro": 0.7148902330358093,
        "f1_true": 0.6835276006157907,
        "f1_false": 0.7462528654558279
    }
}
```

