# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-504
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.7699751257896423,
        "f1_macro": 0.542028708553488,
        "f1_true": 0.6331364183349744,
        "f1_false": 0.4509209987720016
    },
    "test_split_metric": {
        "threshold": 0.8308334350585938,
        "f1_macro": 0.5453123030922781,
        "f1_true": 0.5402722177742194,
        "f1_false": 0.5503523884103367
    }
}
```

