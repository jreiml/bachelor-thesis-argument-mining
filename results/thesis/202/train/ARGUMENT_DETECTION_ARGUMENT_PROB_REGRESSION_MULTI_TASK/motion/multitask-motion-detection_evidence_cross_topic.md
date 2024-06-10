# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 10 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-detection_evidence_cross_topic \
    --run_name multitask-motion-detection_evidence_cross_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.38685324788093567,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.6019600033760071,
    "test_loss": 0.9888132512569427,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7281729991915926,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7021322972733155,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7494434289881348,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7281729991915926,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7142536442766504,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8523045598136514,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7281729991915926,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8002754662774849,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7281729991915926,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7902043362969896,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.6140602582496413,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9712423312883436,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.45726495726495725,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6660531159610834,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.9344978165938864,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.038646869361400604,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19658806920051575,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.786364032288638,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7899036401510783,
    "epochs_trained_total": 5.656013456686291,
    "epochs_trained_best": 2.6260062477472066
}
```

