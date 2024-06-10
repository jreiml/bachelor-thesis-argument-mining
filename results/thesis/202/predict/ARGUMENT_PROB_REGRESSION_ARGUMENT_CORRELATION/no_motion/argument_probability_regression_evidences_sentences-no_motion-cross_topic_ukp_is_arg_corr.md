# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 202 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.23845486342906952,
        "f1_macro": 0.7038307753067854,
        "f1_true": 0.6646776964108128,
        "f1_false": 0.7429838542027579
    },
    "test_split_metric": {
        "threshold": 0.20560109615325928,
        "f1_macro": 0.7418203944635093,
        "f1_true": 0.7330543933054393,
        "f1_false": 0.7505863956215794
    }
}
```

