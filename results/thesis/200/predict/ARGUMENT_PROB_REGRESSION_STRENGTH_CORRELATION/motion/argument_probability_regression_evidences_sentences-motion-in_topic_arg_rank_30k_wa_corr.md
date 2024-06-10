# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_probability_regression_evidences_sentences-motion-in_topic_arg_rank_30k_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_probability_regression_evidences_sentences-motion-in_topic/checkpoint-957
```

# Results
```json
{
    "all_splits_metric": {
        "mean_squared_error": 0.1295531541109085,
        "root_mean_squared_error": 0.35993492603302,
        "spearman": 0.2955372603498664,
        "pearson": 0.3093203324469792
    },
    "test_split_metric": {
        "mean_squared_error": 0.13367046415805817,
        "root_mean_squared_error": 0.36560970544815063,
        "spearman": 0.29196027544241704,
        "pearson": 0.30514039636860385
    }
}
```

