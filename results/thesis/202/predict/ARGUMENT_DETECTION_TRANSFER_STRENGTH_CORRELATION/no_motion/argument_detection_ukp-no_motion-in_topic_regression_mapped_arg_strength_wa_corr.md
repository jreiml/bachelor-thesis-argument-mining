# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03535036742687225,
            "root_mean_squared_error": 0.18801693618297577,
            "spearman": 0.15375875936456126,
            "pearson": 0.159717005657357
        },
        "positive_output": {
            "mean_squared_error": 0.03553932160139084,
            "root_mean_squared_error": 0.18851876258850098,
            "spearman": 0.15301425155618292,
            "pearson": 0.14305348375427038
        },
        "negative_output": {
            "mean_squared_error": 0.035529837012290955,
            "root_mean_squared_error": 0.18849359452724457,
            "spearman": 0.1507679931159789,
            "pearson": 0.14473132689893228
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.03555738553404808,
            "root_mean_squared_error": 0.188566654920578,
            "spearman": 0.15233478769257977,
            "pearson": 0.14168856202735677
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.035520415753126144,
            "root_mean_squared_error": 0.18846860527992249,
            "spearman": 0.15233593418360164,
            "pearson": 0.14484943788326748
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.1058164834976196,
            "root_mean_squared_error": 1.0515780448913574,
            "spearman": 0.1530159725877811,
            "pearson": 0.15241660008526353
        },
        "negative_output": {
            "mean_squared_error": 1.7994388341903687,
            "root_mean_squared_error": 1.3414316177368164,
            "spearman": -0.1507677260376312,
            "pearson": -0.14020347781716536
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.12133321911096573,
            "root_mean_squared_error": 0.34832918643951416,
            "spearman": 0.15233428951356828,
            "pearson": 0.14422818458371223
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.36139199137687683,
            "root_mean_squared_error": 0.601158857345581,
            "spearman": -0.15233432424040216,
            "pearson": -0.14422818355189834
        }
    }
}
```

