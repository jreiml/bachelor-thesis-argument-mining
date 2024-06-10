# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-336
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03647437319159508,
            "root_mean_squared_error": 0.1909826546907425,
            "spearman": 0.23871698059135307,
            "pearson": 0.28138653420380977
        },
        "positive_output": {
            "mean_squared_error": 0.03647559508681297,
            "root_mean_squared_error": 0.19098584353923798,
            "spearman": 0.23876554858419116,
            "pearson": 0.28174402122893405
        },
        "negative_output": {
            "mean_squared_error": 0.036670807749032974,
            "root_mean_squared_error": 0.19149623811244965,
            "spearman": 0.23714113918207053,
            "pearson": 0.2730381711078203
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.03651271387934685,
            "root_mean_squared_error": 0.19108299911022186,
            "spearman": 0.23867010807010647,
            "pearson": 0.28012777791481475
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.036562688648700714,
            "root_mean_squared_error": 0.1912137269973755,
            "spearman": 0.2386701049015613,
            "pearson": 0.2780120778393754
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 0.783794105052948,
            "root_mean_squared_error": 0.8853214979171753,
            "spearman": 0.23876592651385178,
            "pearson": 0.28068227025031045
        },
        "negative_output": {
            "mean_squared_error": 1.468048095703125,
            "root_mean_squared_error": 1.2116303443908691,
            "spearman": -0.23714052410893685,
            "pearson": -0.27306080312116465
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08421023190021515,
            "root_mean_squared_error": 0.2901899814605713,
            "spearman": 0.23866996893497452,
            "pearson": 0.2789566111359117
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.3272646367549896,
            "root_mean_squared_error": 0.5720704793930054,
            "spearman": -0.23867031858588297,
            "pearson": -0.2789566149068859
        }
    }
}
```

