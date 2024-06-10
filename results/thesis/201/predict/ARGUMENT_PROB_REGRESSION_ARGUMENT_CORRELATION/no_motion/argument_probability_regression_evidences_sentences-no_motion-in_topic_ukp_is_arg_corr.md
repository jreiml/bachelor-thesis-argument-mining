# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name argument_probability_regression_evidences_sentences-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-no_motion-in_topic/checkpoint-726
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.16164615750312805,
        "f1_macro": 0.7187124545753214,
        "f1_true": 0.6906079755938113,
        "f1_false": 0.7468169335568315
    },
    "test_split_metric": {
        "threshold": 0.16110678017139435,
        "f1_macro": 0.7308774852115232,
        "f1_true": 0.7131782945736435,
        "f1_false": 0.748576675849403
    }
}
```

