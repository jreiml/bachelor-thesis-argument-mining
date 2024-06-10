# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03521833196282387,
            "root_mean_squared_error": 0.18766547739505768,
            "spearman": 0.171142288243676,
            "pearson": 0.17082209067659107
        },
        "positive_output": {
            "mean_squared_error": 0.03542359918355942,
            "root_mean_squared_error": 0.1882115751504898,
            "spearman": 0.15643469328282672,
            "pearson": 0.15407730709101689
        },
        "negative_output": {
            "mean_squared_error": 0.03521877899765968,
            "root_mean_squared_error": 0.18766666948795319,
            "spearman": 0.17173701933835306,
            "pearson": 0.1715822939498995
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.03537604957818985,
            "root_mean_squared_error": 0.18808521330356598,
            "spearman": 0.16503845889607968,
            "pearson": 0.1590783619180964
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.035343363881111145,
            "root_mean_squared_error": 0.18799830973148346,
            "spearman": 0.16503784387850964,
            "pearson": 0.16160748669866543
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.2907025814056396,
            "root_mean_squared_error": 1.136090874671936,
            "spearman": 0.1563921954159257,
            "pearson": 0.1578839739816293
        },
        "negative_output": {
            "mean_squared_error": 2.455226421356201,
            "root_mean_squared_error": 1.5669162273406982,
            "spearman": -0.17173707148927828,
            "pearson": -0.16988165639524228
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.1255500614643097,
            "root_mean_squared_error": 0.354330450296402,
            "spearman": 0.1650381112471127,
            "pearson": 0.16108464760454225
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.33991876244544983,
            "root_mean_squared_error": 0.5830255150794983,
            "spearman": -0.1650381112471127,
            "pearson": -0.16108464468300707
        }
    }
}
```

