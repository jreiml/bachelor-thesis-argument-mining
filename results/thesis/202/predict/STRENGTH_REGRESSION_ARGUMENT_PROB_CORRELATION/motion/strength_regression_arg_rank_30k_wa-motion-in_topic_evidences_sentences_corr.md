# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1564
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.2643032670021057,
        "root_mean_squared_error": 0.5141043066978455,
        "spearman": 0.3870502527700824,
        "pearson": 0.3558577233545319
    },
    "test_split_metric": {
        "mean_squared_error": 0.2679741680622101,
        "root_mean_squared_error": 0.5176622271537781,
        "spearman": 0.39148266148918137,
        "pearson": 0.35147155818431347
    }
}
```

