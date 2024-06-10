# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-1386
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.23615415394306183,
        "root_mean_squared_error": 0.48595693707466125,
        "spearman": 0.4020745067893256,
        "pearson": 0.3683677591594011
    },
    "test_split_metric": {
        "mean_squared_error": 0.23880396783351898,
        "root_mean_squared_error": 0.4886757433414459,
        "spearman": 0.46877588100834594,
        "pearson": 0.427207095272527
    }
}
```

