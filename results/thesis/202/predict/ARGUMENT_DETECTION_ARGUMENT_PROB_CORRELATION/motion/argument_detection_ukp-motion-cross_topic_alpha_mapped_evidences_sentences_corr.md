# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-336
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.4736842215061188,
        "f1_macro": 0.6749411002992869,
        "f1_true": 0.5228883355678989,
        "f1_false": 0.8269938650306747
    },
    "test_split_metric": {
        "threshold": 0.4736842215061188,
        "f1_macro": 0.7029813590473808,
        "f1_true": 0.562192118226601,
        "f1_false": 0.8437705998681607
    }
}
```

