# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-1089
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.2754523456096649,
        "root_mean_squared_error": 0.5248355269432068,
        "spearman": 0.36214855562896997,
        "pearson": 0.330689941258501
    },
    "test_split_metric": {
        "mean_squared_error": 0.2803356647491455,
        "root_mean_squared_error": 0.5294673442840576,
        "spearman": 0.4210513944408806,
        "pearson": 0.3815464163403637
    }
}
```

