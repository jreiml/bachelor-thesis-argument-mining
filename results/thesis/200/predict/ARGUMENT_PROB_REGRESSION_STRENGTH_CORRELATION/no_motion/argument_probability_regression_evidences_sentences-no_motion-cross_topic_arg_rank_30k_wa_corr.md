# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_arg_rank_30k_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-561
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.13537883758544922,
        "root_mean_squared_error": 0.3679386377334595,
        "spearman": 0.22128355706603955,
        "pearson": 0.22505109998717718
    },
    "test_split_metric": {
        "mean_squared_error": 0.152284637093544,
        "root_mean_squared_error": 0.39023664593696594,
        "spearman": 0.20527279541444332,
        "pearson": 0.20663682524782814
    }
}
```

