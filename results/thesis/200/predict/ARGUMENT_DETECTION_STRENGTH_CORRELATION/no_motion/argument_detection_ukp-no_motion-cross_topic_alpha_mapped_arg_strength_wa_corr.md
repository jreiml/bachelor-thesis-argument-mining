# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-280
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.7407018542289734,
        "f1_macro": 0.5522000828188178,
        "f1_true": 0.7187686718768672,
        "f1_false": 0.3856314937607685
    },
    "test_split_metric": {
        "threshold": 0.7630389332771301,
        "f1_macro": 0.5481769069504918,
        "f1_true": 0.6651572327044025,
        "f1_false": 0.4311965811965812
    }
}
```

