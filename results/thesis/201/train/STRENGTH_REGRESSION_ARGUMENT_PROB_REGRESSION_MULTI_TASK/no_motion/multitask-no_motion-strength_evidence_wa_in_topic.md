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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.037410322576761246,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.040536168962717056,
    "test_loss": 0.0779464915394783,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.026721661910414696,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1634676158428192,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.48908824934073797,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5367283508380302,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.040584854781627655,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.20145682990550995,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7791542868920031,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8008374897544294,
    "epochs_trained_total": 6.328767123287671,
    "epochs_trained_best": 3.315068493150685
}
```

