# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-476
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.4285714328289032,
        "f1_macro": 0.6643086902521977,
        "f1_true": 0.5246435845213849,
        "f1_false": 0.8039737959830106
    },
    "test_split_metric": {
        "threshold": 0.4545454680919647,
        "f1_macro": 0.7071851727773771,
        "f1_true": 0.5796269727403156,
        "f1_false": 0.8347433728144387
    }
}
```

