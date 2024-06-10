# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1666
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.6869374513626099,
        "f1_macro": 0.6980773958304215,
        "f1_true": 0.661882688536281,
        "f1_false": 0.734272103124562
    },
    "test_split_metric": {
        "threshold": 0.6943554282188416,
        "f1_macro": 0.7189954285277615,
        "f1_true": 0.6984797297297297,
        "f1_false": 0.7395111273257935
    }
}
```

