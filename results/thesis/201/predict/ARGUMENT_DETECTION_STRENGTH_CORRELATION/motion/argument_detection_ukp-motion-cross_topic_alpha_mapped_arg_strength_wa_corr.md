# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-392
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.613389790058136,
        "f1_macro": 0.5965471457840201,
        "f1_true": 0.8669532952470699,
        "f1_false": 0.32614099632097043
    },
    "test_split_metric": {
        "threshold": 0.6671284437179565,
        "f1_macro": 0.6092105652129123,
        "f1_true": 0.8233248081841432,
        "f1_false": 0.3950963222416813
    }
}
```

