# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-280
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.517241358757019,
        "f1_macro": 0.663467063462223,
        "f1_true": 0.488116353316779,
        "f1_false": 0.838817773607667
    },
    "test_split_metric": {
        "threshold": 0.4736842215061188,
        "f1_macro": 0.6879953686377647,
        "f1_true": 0.527369826435247,
        "f1_false": 0.8486209108402822
    }
}
```

