# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-759
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.12397190183401108,
        "root_mean_squared_error": 0.35209643840789795,
        "spearman": 0.28103283232151755,
        "pearson": 0.28685039979737686
    },
    "test_split_metric": {
        "mean_squared_error": 0.12885499000549316,
        "root_mean_squared_error": 0.3589637577533722,
        "spearman": 0.2740633791856743,
        "pearson": 0.2775864588663283
    }
}
```

