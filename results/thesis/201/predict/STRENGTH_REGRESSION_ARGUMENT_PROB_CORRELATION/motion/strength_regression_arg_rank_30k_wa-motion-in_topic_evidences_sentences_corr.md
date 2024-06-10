# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-1802
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.3193322420120239,
        "root_mean_squared_error": 0.5650948882102966,
        "spearman": 0.35885198901445436,
        "pearson": 0.32473779781950945
    },
    "test_split_metric": {
        "mean_squared_error": 0.3225845992565155,
        "root_mean_squared_error": 0.5679653286933899,
        "spearman": 0.36191480770926737,
        "pearson": 0.3237213611444467
    }
}
```

