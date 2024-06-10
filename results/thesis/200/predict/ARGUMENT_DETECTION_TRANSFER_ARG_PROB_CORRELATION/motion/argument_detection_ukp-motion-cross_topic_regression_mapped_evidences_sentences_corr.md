# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-476
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.0711313858628273,
            "root_mean_squared_error": 0.2667046785354614,
            "spearman": 0.5764825104710183,
            "pearson": 0.5657131185980921
        },
        "positive_output": {
            "mean_squared_error": 0.08029922842979431,
            "root_mean_squared_error": 0.2833711802959442,
            "spearman": 0.48728696391544724,
            "pearson": 0.49696428539867804
        },
        "negative_output": {
            "mean_squared_error": 0.07147073745727539,
            "root_mean_squared_error": 0.26734012365341187,
            "spearman": 0.5687120193666577,
            "pearson": 0.5586186397568578
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.07478383928537369,
            "root_mean_squared_error": 0.2734663486480713,
            "spearman": 0.5335378323082868,
            "pearson": 0.5367868911232757
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.07863373309373856,
            "root_mean_squared_error": 0.2804170846939087,
            "spearman": 0.5335380318033793,
            "pearson": 0.5361448685172333
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.5741727352142334,
            "root_mean_squared_error": 1.6044228076934814,
            "spearman": 0.4872868523193052,
            "pearson": 0.4932393952999441
        },
        "negative_output": {
            "mean_squared_error": 1.1946848630905151,
            "root_mean_squared_error": 1.0930163860321045,
            "spearman": -0.5687121903908379,
            "pearson": -0.5548461385623075
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.09410955011844635,
            "root_mean_squared_error": 0.30677279829978943,
            "spearman": 0.5335378181938366,
            "pearson": 0.5366113999567865
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.4572279751300812,
            "root_mean_squared_error": 0.6761863231658936,
            "spearman": -0.5335378182006345,
            "pearson": -0.5366114027391157
        }
    }
}
```

