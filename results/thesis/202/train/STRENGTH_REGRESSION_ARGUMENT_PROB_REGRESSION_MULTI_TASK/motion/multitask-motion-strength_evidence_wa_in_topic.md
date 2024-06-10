# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1.6 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_evidence_wa_in_topic \
    --run_name multitask-motion-strength_evidence_wa_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.0337173268198967,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.04034142941236496,
    "test_loss": 0.07405875623226166,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025213392451405525,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.15878725051879883,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.5060209579607698,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5546116341042588,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03375464677810669,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.18372437357902527,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.81383385232313,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8344483071244779,
    "epochs_trained_total": 5.6255707762557075,
    "epochs_trained_best": 2.6118721461187215
}
```

