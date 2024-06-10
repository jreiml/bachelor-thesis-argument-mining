# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 201 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-759
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.25531527400016785,
        "f1_macro": 0.7105750051324964,
        "f1_true": 0.6649904549052476,
        "f1_false": 0.7561595553597451
    },
    "test_split_metric": {
        "threshold": 0.270413339138031,
        "f1_macro": 0.715038778194345,
        "f1_true": 0.6770551038843722,
        "f1_false": 0.7530224525043179
    }
}
```

