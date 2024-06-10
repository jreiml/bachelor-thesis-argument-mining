# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-280
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.07966645061969757,
            "root_mean_squared_error": 0.2822524607181549,
            "spearman": 0.49500557430026726,
            "pearson": 0.5083746175693323
        },
        "positive_output": {
            "mean_squared_error": 0.0774170383810997,
            "root_mean_squared_error": 0.2782391607761383,
            "spearman": 0.49590717535559403,
            "pearson": 0.5092123915334507
        },
        "negative_output": {
            "mean_squared_error": 0.08085863292217255,
            "root_mean_squared_error": 0.2843565344810486,
            "spearman": 0.45919493926311433,
            "pearson": 0.47720720497213603
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.07777351886034012,
            "root_mean_squared_error": 0.2788790464401245,
            "spearman": 0.4861765830120683,
            "pearson": 0.5001615144335602
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.08195946365594864,
            "root_mean_squared_error": 0.2862856388092041,
            "spearman": 0.4861764811445823,
            "pearson": 0.4980483517781131
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.363657236099243,
            "root_mean_squared_error": 1.5374189615249634,
            "spearman": 0.49590741811959227,
            "pearson": 0.5075802507420465
        },
        "negative_output": {
            "mean_squared_error": 0.6693040728569031,
            "root_mean_squared_error": 0.8181100487709045,
            "spearman": -0.45919450811558193,
            "pearson": -0.4738034249482549
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08905680477619171,
            "root_mean_squared_error": 0.2984238564968109,
            "spearman": 0.48617648518615136,
            "pearson": 0.4991249169200065
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.4686789810657501,
            "root_mean_squared_error": 0.6846013069152832,
            "spearman": -0.4861765431712143,
            "pearson": -0.4991249158489252
        }
    }
}
```

