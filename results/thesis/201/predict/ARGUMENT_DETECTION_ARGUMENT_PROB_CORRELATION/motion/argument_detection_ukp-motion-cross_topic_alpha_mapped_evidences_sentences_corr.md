# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-392
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.4285714328289032,
        "f1_macro": 0.6864679054247869,
        "f1_true": 0.566409691629956,
        "f1_false": 0.8065261192196176
    },
    "test_split_metric": {
        "threshold": 0.3478260934352875,
        "f1_macro": 0.7233511825078089,
        "f1_true": 0.6281481481481481,
        "f1_false": 0.8185542168674699
    }
}
```

