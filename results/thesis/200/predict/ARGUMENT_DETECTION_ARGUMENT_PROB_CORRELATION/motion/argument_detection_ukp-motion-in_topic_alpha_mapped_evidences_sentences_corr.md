# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-580
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.1785714328289032,
        "f1_macro": 0.7113055819321503,
        "f1_true": 0.687869822485207,
        "f1_false": 0.7347413413790935
    },
    "test_split_metric": {
        "threshold": 0.1818181872367859,
        "f1_macro": 0.7083784806818569,
        "f1_true": 0.6806026365348401,
        "f1_false": 0.7361543248288737
    }
}
```

