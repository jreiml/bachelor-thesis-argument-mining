# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-580
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.4884249269962311,
        "f1_macro": 0.5963459630413421,
        "f1_true": 0.9385458998065696,
        "f1_false": 0.2541460262761146
    },
    "test_split_metric": {
        "threshold": 0.4913514256477356,
        "f1_macro": 0.5818591618140498,
        "f1_true": 0.9373032292884771,
        "f1_false": 0.22641509433962262
    }
}
```

