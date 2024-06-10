# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 9.25 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-detection_evidence_in_topic \
    --run_name multitask-motion-detection_evidence_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.33261194825172424,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4298647940158844,
    "test_loss": 0.7624767422676086,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8064200430612645,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8062745739147585,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8059451902044557,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8064200430612645,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8143373716356803,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8185413802756966,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8064200430612645,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8112434903485167,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8064200430612645,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8115831586968947,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8009659891326223,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7465825446898002,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8820921985815603,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8889816360601002,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7335053446369333,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.035958047956228256,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.1896260678768158,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7987588546324363,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8141175708861924,
    "epochs_trained_total": 4.807219031993437,
    "epochs_trained_best": 1.802707136997539
}
```

