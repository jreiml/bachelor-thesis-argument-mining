# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-726
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.16763058304786682,
        "root_mean_squared_error": 0.40942713618278503,
        "spearman": 0.25568847014356133,
        "pearson": 0.2643649882050079
    },
    "test_split_metric": {
        "mean_squared_error": 0.17137250304222107,
        "root_mean_squared_error": 0.4139716327190399,
        "spearman": 0.24700807214216866,
        "pearson": 0.25467266424157653
    }
}
```

