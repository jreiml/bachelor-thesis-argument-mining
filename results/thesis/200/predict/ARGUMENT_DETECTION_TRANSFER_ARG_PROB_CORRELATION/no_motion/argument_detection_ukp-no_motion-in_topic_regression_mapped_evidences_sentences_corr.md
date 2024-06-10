# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.08144672960042953,
            "root_mean_squared_error": 0.2853887379169464,
            "spearman": 0.4969353626665546,
            "pearson": 0.4857830741666426
        },
        "positive_output": {
            "mean_squared_error": 0.08255469053983688,
            "root_mean_squared_error": 0.28732332587242126,
            "spearman": 0.4785040375196873,
            "pearson": 0.474921037980485
        },
        "negative_output": {
            "mean_squared_error": 0.08205188810825348,
            "root_mean_squared_error": 0.28644701838493347,
            "spearman": 0.49685875657741235,
            "pearson": 0.4800425410379884
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08422486484050751,
            "root_mean_squared_error": 0.2902151942253113,
            "spearman": 0.49793682249878807,
            "pearson": 0.4585546270417925
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.08459862321615219,
            "root_mean_squared_error": 0.2908584177494049,
            "spearman": 0.49793688797847013,
            "pearson": 0.45474553499555215
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.6607258319854736,
            "root_mean_squared_error": 1.6311731338500977,
            "spearman": 0.47850422289437833,
            "pearson": 0.47234015776302934
        },
        "negative_output": {
            "mean_squared_error": 1.6755167245864868,
            "root_mean_squared_error": 1.2944175004959106,
            "spearman": -0.49685872065378434,
            "pearson": -0.48245305761172114
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.1016324907541275,
            "root_mean_squared_error": 0.3187985122203827,
            "spearman": 0.4979368742414399,
            "pearson": 0.45688715244655503
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.49440649151802063,
            "root_mean_squared_error": 0.7031404376029968,
            "spearman": -0.4979368641757833,
            "pearson": -0.45688715419964915
        }
    }
}
```

