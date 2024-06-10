# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-336
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6524443626403809,
        "f1_macro": 0.6044433886419746,
        "f1_true": 0.8304257095158597,
        "f1_false": 0.37846106776808935
    },
    "test_split_metric": {
        "threshold": 0.7014287710189819,
        "f1_macro": 0.6084534850492297,
        "f1_true": 0.7851609383524277,
        "f1_false": 0.4317460317460317
    }
}
```

