# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03292286396026611,
            "root_mean_squared_error": 0.18144658207893372,
            "spearman": 0.29228005302142696,
            "pearson": 0.3046670035354795
        },
        "positive_output": {
            "mean_squared_error": 0.03296739608049393,
            "root_mean_squared_error": 0.18156926333904266,
            "spearman": 0.29238542455068467,
            "pearson": 0.30211910256400454
        },
        "negative_output": {
            "mean_squared_error": 0.03280645236372948,
            "root_mean_squared_error": 0.18112552165985107,
            "spearman": 0.26967732571186076,
            "pearson": 0.30962046597920073
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.03319304436445236,
            "root_mean_squared_error": 0.18218958377838135,
            "spearman": 0.2813093737822048,
            "pearson": 0.29216560664918995
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.03312088921666145,
            "root_mean_squared_error": 0.18199145793914795,
            "spearman": 0.28130925875779994,
            "pearson": 0.29547021628984044
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 0.7508195042610168,
            "root_mean_squared_error": 0.8664984107017517,
            "spearman": 0.2923851712522044,
            "pearson": 0.30974997379484326
        },
        "negative_output": {
            "mean_squared_error": 2.673738956451416,
            "root_mean_squared_error": 1.6351571083068848,
            "spearman": -0.26967716218089444,
            "pearson": -0.30992893193959037
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.04978816583752632,
            "root_mean_squared_error": 0.22313262522220612,
            "spearman": 0.2813075655290733,
            "pearson": 0.29434042200044946
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.5007697343826294,
            "root_mean_squared_error": 0.7076508402824402,
            "spearman": -0.28130652261177364,
            "pearson": -0.2943404184948863
        }
    }
}
```

