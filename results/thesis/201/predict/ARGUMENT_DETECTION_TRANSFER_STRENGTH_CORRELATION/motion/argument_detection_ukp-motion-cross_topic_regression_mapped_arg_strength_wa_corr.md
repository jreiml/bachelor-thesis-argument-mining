# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-392
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03577771782875061,
            "root_mean_squared_error": 0.18914999067783356,
            "spearman": 0.26995239055898146,
            "pearson": 0.3109753382003415
        },
        "positive_output": {
            "mean_squared_error": 0.035814810544252396,
            "root_mean_squared_error": 0.1892480105161667,
            "spearman": 0.2706461243612126,
            "pearson": 0.3103726304663629
        },
        "negative_output": {
            "mean_squared_error": 0.036351997405290604,
            "root_mean_squared_error": 0.19066199660301208,
            "spearman": 0.2633017552796638,
            "pearson": 0.30461001072423655
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.036042191088199615,
            "root_mean_squared_error": 0.18984781205654144,
            "spearman": 0.2670856106528193,
            "pearson": 0.30364854703669625
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.03605668619275093,
            "root_mean_squared_error": 0.18988598883152008,
            "spearman": 0.26708574134585666,
            "pearson": 0.3030831774051542
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 0.7858349680900574,
            "root_mean_squared_error": 0.8864733576774597,
            "spearman": 0.2706463159077118,
            "pearson": 0.3123152444877182
        },
        "negative_output": {
            "mean_squared_error": 4.126828670501709,
            "root_mean_squared_error": 2.0314598083496094,
            "spearman": -0.2633014166194514,
            "pearson": -0.3044871660276895
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.07802378386259079,
            "root_mean_squared_error": 0.2793273627758026,
            "spearman": 0.26708595750969794,
            "pearson": 0.30351197423768994
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.38462603092193604,
            "root_mean_squared_error": 0.6201822757720947,
            "spearman": -0.26708582426439104,
            "pearson": -0.30351197468751784
        }
    }
}
```

