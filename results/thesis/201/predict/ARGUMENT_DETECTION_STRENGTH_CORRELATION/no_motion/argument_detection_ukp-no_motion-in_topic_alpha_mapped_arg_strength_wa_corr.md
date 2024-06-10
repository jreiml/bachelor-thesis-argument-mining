# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-522
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.7000203132629395,
        "f1_macro": 0.5472001316821691,
        "f1_true": 0.7599330730619075,
        "f1_false": 0.3344671903024306
    },
    "test_split_metric": {
        "threshold": 0.707631528377533,
        "f1_macro": 0.5527840178516543,
        "f1_true": 0.7613210759278175,
        "f1_false": 0.3442469597754911
    }
}
```

