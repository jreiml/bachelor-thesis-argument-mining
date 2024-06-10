# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-660
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.32500964403152466,
        "root_mean_squared_error": 0.5700961947441101,
        "spearman": 0.36417458390219076,
        "pearson": 0.33517124277136023
    },
    "test_split_metric": {
        "mean_squared_error": 0.33232054114341736,
        "root_mean_squared_error": 0.5764725208282471,
        "spearman": 0.3939855242848358,
        "pearson": 0.3644448471184944
    }
}
```

