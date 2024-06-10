# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-280
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.038949549198150635,
            "root_mean_squared_error": 0.19735640287399292,
            "spearman": 0.1372341125234359,
            "pearson": 0.13726698955375802
        },
        "positive_output": {
            "mean_squared_error": 0.039170753210783005,
            "root_mean_squared_error": 0.19791603088378906,
            "spearman": 0.12719035695188777,
            "pearson": 0.11880453331883208
        },
        "negative_output": {
            "mean_squared_error": 0.038902632892131805,
            "root_mean_squared_error": 0.1972375065088272,
            "spearman": 0.1386848831238296,
            "pearson": 0.13891198814328493
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.039134204387664795,
            "root_mean_squared_error": 0.1978236734867096,
            "spearman": 0.1320863987939677,
            "pearson": 0.12231045069032198
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.03912857547402382,
            "root_mean_squared_error": 0.19780944287776947,
            "spearman": 0.1320862319993574,
            "pearson": 0.12345695775682795
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.1345279216766357,
            "root_mean_squared_error": 1.0651421546936035,
            "spearman": 0.12719008256083142,
            "pearson": 0.1239897346075378
        },
        "negative_output": {
            "mean_squared_error": 1.1313631534576416,
            "root_mean_squared_error": 1.0636556148529053,
            "spearman": -0.13868454299564478,
            "pearson": -0.13873300952632459
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.14135678112506866,
            "root_mean_squared_error": 0.3759744465351105,
            "spearman": 0.13208580175720513,
            "pearson": 0.1232982309279799
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.27158161997795105,
            "root_mean_squared_error": 0.5211349129676819,
            "spearman": -0.1320858814015418,
            "pearson": -0.12329822987278637
        }
    }
}
```

