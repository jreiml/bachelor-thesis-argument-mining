# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-884
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.24476145207881927,
        "root_mean_squared_error": 0.4947337210178375,
        "spearman": 0.3518028902271265,
        "pearson": 0.3245669851284187
    },
    "test_split_metric": {
        "mean_squared_error": 0.24793748557567596,
        "root_mean_squared_error": 0.49793320894241333,
        "spearman": 0.35319355251949425,
        "pearson": 0.31487979579495784
    }
}
```

