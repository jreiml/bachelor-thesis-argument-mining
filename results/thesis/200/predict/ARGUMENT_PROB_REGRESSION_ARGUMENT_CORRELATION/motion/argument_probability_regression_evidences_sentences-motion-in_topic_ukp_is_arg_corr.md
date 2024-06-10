# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_ukp_is_arg_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "threshold": 0.2009638547897339,
        "f1_macro": 0.7214502958190026,
        "f1_true": 0.6852711780352554,
        "f1_false": 0.7576294136027497
    },
    "test_split_metric": {
        "threshold": 0.21083374321460724,
        "f1_macro": 0.7267880738762733,
        "f1_true": 0.7005845421086816,
        "f1_false": 0.7529916056438651
    }
}
```

