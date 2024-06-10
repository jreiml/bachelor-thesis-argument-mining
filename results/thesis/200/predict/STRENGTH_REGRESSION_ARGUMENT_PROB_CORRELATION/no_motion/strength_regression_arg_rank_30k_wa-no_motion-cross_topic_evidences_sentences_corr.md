# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.24362020194530487,
        "root_mean_squared_error": 0.4935789704322815,
        "spearman": 0.38622609805825114,
        "pearson": 0.35434377179954285
    },
    "test_split_metric": {
        "mean_squared_error": 0.2508391737937927,
        "root_mean_squared_error": 0.5008384585380554,
        "spearman": 0.4408590995986669,
        "pearson": 0.4018940938699269
    }
}
```

