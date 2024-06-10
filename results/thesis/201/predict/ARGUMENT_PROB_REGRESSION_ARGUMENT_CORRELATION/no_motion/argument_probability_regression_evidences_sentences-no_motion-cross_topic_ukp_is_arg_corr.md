# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-396
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.2823738753795624,
        "f1_macro": 0.7085344904515438,
        "f1_true": 0.6846153846153845,
        "f1_false": 0.7324535962877031
    },
    "test_split_metric": {
        "threshold": 0.2710382044315338,
        "f1_macro": 0.75323353903645,
        "f1_true": 0.7530839231547016,
        "f1_false": 0.7533831549181983
    }
}
```

