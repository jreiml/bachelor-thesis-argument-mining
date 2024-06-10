# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 7.6 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_cross_topic \
    --run_name multitask-no_motion-detection_evidence_cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5934270024299622,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4065987169742584,
    "test_loss": 1.0000257194042206,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.6968472109943411,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.6643315503377851,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.723704302134042,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.6968472109943411,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.6819903256252949,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8378767497214065,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.6968472109943411,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7680209660604657,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.6968472109943411,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7688039457459926,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.5598591549295775,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9562883435582822,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.4076923076923077,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6427835051546392,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8932584269662921,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05349982902407646,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.23130029439926147,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.6819694796067876,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.6905972066331606,
    "epochs_trained_total": 5.1513877207737595,
    "epochs_trained_best": 2.1211596497303713
}
```

