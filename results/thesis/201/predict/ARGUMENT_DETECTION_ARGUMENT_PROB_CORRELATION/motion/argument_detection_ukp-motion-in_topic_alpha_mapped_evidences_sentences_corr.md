# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-522
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.26923078298568726,
        "f1_macro": 0.7048831429190239,
        "f1_true": 0.6362307930744365,
        "f1_false": 0.7735354927636113
    },
    "test_split_metric": {
        "threshold": 0.30434781312942505,
        "f1_macro": 0.702223870008367,
        "f1_true": 0.6198775317946302,
        "f1_false": 0.7845702082221035
    }
}
```

