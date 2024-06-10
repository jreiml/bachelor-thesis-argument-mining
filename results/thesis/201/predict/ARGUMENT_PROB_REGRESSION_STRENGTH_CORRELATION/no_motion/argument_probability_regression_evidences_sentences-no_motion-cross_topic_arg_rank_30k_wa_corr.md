# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-396
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.11869527399539948,
        "root_mean_squared_error": 0.34452179074287415,
        "spearman": 0.18496893318776805,
        "pearson": 0.19125780875705742
    },
    "test_split_metric": {
        "mean_squared_error": 0.13760381937026978,
        "root_mean_squared_error": 0.3709498941898346,
        "spearman": 0.14789669777884643,
        "pearson": 0.15177886069529034
    }
}
```

