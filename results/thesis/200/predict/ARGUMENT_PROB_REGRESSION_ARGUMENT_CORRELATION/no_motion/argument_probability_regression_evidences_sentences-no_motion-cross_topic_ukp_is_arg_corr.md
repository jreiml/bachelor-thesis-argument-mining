# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-cross_topic/checkpoint-561
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.24931439757347107,
        "f1_macro": 0.7094195410413072,
        "f1_true": 0.6714498443211045,
        "f1_false": 0.7473892377615099
    },
    "test_split_metric": {
        "threshold": 0.22988882660865784,
        "f1_macro": 0.7552387149309558,
        "f1_true": 0.7450980392156863,
        "f1_false": 0.7653793906462254
    }
}
```

