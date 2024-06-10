# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.5600000023841858,
        "f1_macro": 0.658357327791683,
        "f1_true": 0.46697056323881636,
        "f1_false": 0.8497440923445496
    },
    "test_split_metric": {
        "threshold": 0.47826087474823,
        "f1_macro": 0.6488169310357241,
        "f1_true": 0.46170678336980314,
        "f1_false": 0.8359270787016452
    }
}
```

