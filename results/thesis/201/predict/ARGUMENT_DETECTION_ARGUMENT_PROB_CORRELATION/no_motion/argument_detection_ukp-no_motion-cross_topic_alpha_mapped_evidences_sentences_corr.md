# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-504
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.5882353186607361,
        "f1_macro": 0.6341416816367056,
        "f1_true": 0.41305281236582225,
        "f1_false": 0.855230550907589
    },
    "test_split_metric": {
        "threshold": 0.5882353186607361,
        "f1_macro": 0.6443189782737253,
        "f1_true": 0.42209753992231336,
        "f1_false": 0.8665404166251371
    }
}
```

