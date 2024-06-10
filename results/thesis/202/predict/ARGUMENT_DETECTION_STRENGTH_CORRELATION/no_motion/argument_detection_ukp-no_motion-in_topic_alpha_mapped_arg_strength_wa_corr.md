# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_alpha_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6880508661270142,
        "f1_macro": 0.5535054178719105,
        "f1_true": 0.770215678050288,
        "f1_false": 0.336795157693533
    },
    "test_split_metric": {
        "threshold": 0.744904637336731,
        "f1_macro": 0.5510782864595525,
        "f1_true": 0.7371764705882352,
        "f1_false": 0.3649801023308698
    }
}
```

