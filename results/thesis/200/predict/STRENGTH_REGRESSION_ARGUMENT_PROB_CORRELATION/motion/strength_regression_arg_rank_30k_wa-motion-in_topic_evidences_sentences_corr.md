# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1666
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.2583676874637604,
        "root_mean_squared_error": 0.5082988142967224,
        "spearman": 0.44810165727044526,
        "pearson": 0.4077422702499738
    },
    "test_split_metric": {
        "mean_squared_error": 0.2612815797328949,
        "root_mean_squared_error": 0.5111570954322815,
        "spearman": 0.4505289679387909,
        "pearson": 0.40086898771946966
    }
}
```

