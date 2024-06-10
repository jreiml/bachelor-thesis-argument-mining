# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-522
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.08428934961557388,
            "root_mean_squared_error": 0.2903262674808502,
            "spearman": 0.46315246614704725,
            "pearson": 0.4574280888967716
        },
        "positive_output": {
            "mean_squared_error": 0.0848916694521904,
            "root_mean_squared_error": 0.2913617491722107,
            "spearman": 0.45898641913238536,
            "pearson": 0.4513196166913455
        },
        "negative_output": {
            "mean_squared_error": 0.08405210822820663,
            "root_mean_squared_error": 0.2899174094200134,
            "spearman": 0.4593602845038976,
            "pearson": 0.4598616983947886
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08639335632324219,
            "root_mean_squared_error": 0.29392746090888977,
            "spearman": 0.4628297901806041,
            "pearson": 0.4356757976404632
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.08667454123497009,
            "root_mean_squared_error": 0.29440540075302124,
            "spearman": 0.4628296388121601,
            "pearson": 0.43264209565147255
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.827648401260376,
            "root_mean_squared_error": 1.6815613508224487,
            "spearman": 0.45898638320931945,
            "pearson": 0.45440210605909226
        },
        "negative_output": {
            "mean_squared_error": 1.6290597915649414,
            "root_mean_squared_error": 1.2763463258743286,
            "spearman": -0.4593602191407182,
            "pearson": -0.4608140316360155
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.10875285416841507,
            "root_mean_squared_error": 0.32977697253227234,
            "spearman": 0.46282973283235773,
            "pearson": 0.4343298000466693
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.5073615908622742,
            "root_mean_squared_error": 0.7122932076454163,
            "spearman": -0.4628297626105445,
            "pearson": -0.4343298022684025
        }
    }
}
```

