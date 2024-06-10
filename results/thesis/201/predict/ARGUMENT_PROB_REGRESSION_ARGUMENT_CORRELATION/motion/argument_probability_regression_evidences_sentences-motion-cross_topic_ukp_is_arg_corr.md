# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-cross_topic/checkpoint-858
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.232413187623024,
        "f1_macro": 0.7119248051050295,
        "f1_true": 0.6777065527065527,
        "f1_false": 0.7461430575035063
    },
    "test_split_metric": {
        "threshold": 0.1964334100484848,
        "f1_macro": 0.7416591223293862,
        "f1_true": 0.7378999179655455,
        "f1_false": 0.745418326693227
    }
}
```

