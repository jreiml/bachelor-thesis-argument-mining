# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 200 \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-364
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.038963738828897476,
            "root_mean_squared_error": 0.19739234447479248,
            "spearman": 0.12286994135573102,
            "pearson": 0.12822849666182118
        },
        "positive_output": {
            "mean_squared_error": 0.038997236639261246,
            "root_mean_squared_error": 0.19747717678546906,
            "spearman": 0.12490247458094006,
            "pearson": 0.13011379747345356
        },
        "negative_output": {
            "mean_squared_error": 0.03903099149465561,
            "root_mean_squared_error": 0.19756262004375458,
            "spearman": 0.12360481206989883,
            "pearson": 0.13173567800001595
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.03914486989378929,
            "root_mean_squared_error": 0.19785062968730927,
            "spearman": 0.12615370528819048,
            "pearson": 0.12435879807379333
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.03909025713801384,
            "root_mean_squared_error": 0.19771255552768707,
            "spearman": 0.1261531310309751,
            "pearson": 0.12505893862632264
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 0.9303541779518127,
            "root_mean_squared_error": 0.9645487070083618,
            "spearman": 0.12490268153302059,
            "pearson": 0.1345576978136013
        },
        "negative_output": {
            "mean_squared_error": 2.1025612354278564,
            "root_mean_squared_error": 1.4500211477279663,
            "spearman": -0.1236049829297105,
            "pearson": -0.13243334530896672
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.11667818576097488,
            "root_mean_squared_error": 0.34158188104629517,
            "spearman": 0.12615269114917976,
            "pearson": 0.1250576734350903
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.32287800312042236,
            "root_mean_squared_error": 0.5682235360145569,
            "spearman": -0.126152732635192,
            "pearson": -0.12505767345713015
        }
    }
}
```

