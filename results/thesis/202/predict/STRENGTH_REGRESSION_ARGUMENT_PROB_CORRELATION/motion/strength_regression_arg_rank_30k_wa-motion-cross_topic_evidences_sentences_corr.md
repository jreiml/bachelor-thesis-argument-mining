# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-1089
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.25761911273002625,
        "root_mean_squared_error": 0.5075619220733643,
        "spearman": 0.38691123334320787,
        "pearson": 0.35903772352694724
    },
    "test_split_metric": {
        "mean_squared_error": 0.26292183995246887,
        "root_mean_squared_error": 0.5127590298652649,
        "spearman": 0.44238595414702014,
        "pearson": 0.4103225384907494
    }
}
```

