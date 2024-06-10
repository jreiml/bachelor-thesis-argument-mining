# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-522
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.4886956512928009,
        "f1_macro": 0.6039630056136792,
        "f1_true": 0.9361603480865937,
        "f1_false": 0.27176566314076483
    },
    "test_split_metric": {
        "threshold": 0.40557876229286194,
        "f1_macro": 0.5898985500968368,
        "f1_true": 0.9510662773205912,
        "f1_false": 0.22873082287308227
    }
}
```

