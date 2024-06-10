# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-522
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.07956356555223465,
            "root_mean_squared_error": 0.282070130109787,
            "spearman": 0.5200663817982989,
            "pearson": 0.5039956700202413
        },
        "positive_output": {
            "mean_squared_error": 0.07974984496831894,
            "root_mean_squared_error": 0.2824001610279083,
            "spearman": 0.5153348237268064,
            "pearson": 0.5024589879614655
        },
        "negative_output": {
            "mean_squared_error": 0.07919109612703323,
            "root_mean_squared_error": 0.2814091145992279,
            "spearman": 0.5222014673976212,
            "pearson": 0.5072692694544257
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08047809451818466,
            "root_mean_squared_error": 0.2836866080760956,
            "spearman": 0.5201376910748385,
            "pearson": 0.49494138214705824
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.08056049048900604,
            "root_mean_squared_error": 0.28383180499076843,
            "spearman": 0.5201376910748385,
            "pearson": 0.49517072399528594
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.589906930923462,
            "root_mean_squared_error": 1.2609151601791382,
            "spearman": 0.5153348495248574,
            "pearson": 0.5018193006908432
        },
        "negative_output": {
            "mean_squared_error": 2.0396814346313477,
            "root_mean_squared_error": 1.4281741380691528,
            "spearman": -0.5222017824815123,
            "pearson": -0.5067536955205864
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.12537376582622528,
            "root_mean_squared_error": 0.35408157110214233,
            "spearman": 0.5201376910748385,
            "pearson": 0.495388087295377
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.4394058287143707,
            "root_mean_squared_error": 0.6628769040107727,
            "spearman": -0.5201376910748385,
            "pearson": -0.4953880882981751
        }
    }
}
```

