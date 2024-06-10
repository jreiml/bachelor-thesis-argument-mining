# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-522
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.5882353186607361,
        "f1_macro": 0.6646361822675901,
        "f1_true": 0.4727099388257726,
        "f1_false": 0.8565624257094077
    },
    "test_split_metric": {
        "threshold": 0.5882353186607361,
        "f1_macro": 0.6626938188070348,
        "f1_true": 0.46636771300448426,
        "f1_false": 0.8590199246095853
    }
}
```

