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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4867863953113556,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.37712353467941284,
    "test_loss": 0.8639099299907684,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7896119644300728,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.781221363145189,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7956819453389515,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7896119644300728,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7812968643490116,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.833111107391445,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7896119644300728,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8165283506756122,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7896119644300728,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8240662497887443,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7383764765016335,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9348159509202454,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.6277777777777778,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7367784829253551,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8962782184258694,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.037698112428188324,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19416001439094543,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7836560291991203,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7937184274390788,
    "epochs_trained_total": 6.968881412952061,
    "epochs_trained_best": 3.9389329725381215
}
```

