# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-476
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.5912368297576904,
        "f1_macro": 0.596611169870145,
        "f1_true": 0.8802941629447097,
        "f1_false": 0.3129281767955801
    },
    "test_split_metric": {
        "threshold": 0.5788513422012329,
        "f1_macro": 0.6131700567090292,
        "f1_true": 0.8758958883440211,
        "f1_false": 0.3504442250740375
    }
}
```

