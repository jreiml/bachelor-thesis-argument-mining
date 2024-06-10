# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1.4 1 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_evidence_wa_in_topic \
    --run_name multitask-no_motion-strength_evidence_wa_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.041371773928403854,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.03804958611726761,
    "test_loss": 0.07942136004567146,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.027095302939414978,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.164606511592865,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.46583293587061997,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5117774009544611,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.041371770203113556,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.20340052247047424,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.768132244390612,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7883816559771347,
    "epochs_trained_total": 5.0228310502283104,
    "epochs_trained_best": 2.009132420091324
}
```

