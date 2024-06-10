# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-924
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.35423052310943604,
        "root_mean_squared_error": 0.5951727032661438,
        "spearman": 0.3267221001939369,
        "pearson": 0.2944537884380876
    },
    "test_split_metric": {
        "mean_squared_error": 0.3634636700153351,
        "root_mean_squared_error": 0.6028794646263123,
        "spearman": 0.35491108239252667,
        "pearson": 0.3213397072272346
    }
}
```

