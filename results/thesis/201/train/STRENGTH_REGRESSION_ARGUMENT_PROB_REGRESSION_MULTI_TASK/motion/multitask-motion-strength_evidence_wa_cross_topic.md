# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1.8 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_evidence_wa_cross_topic \
    --run_name multitask-motion-strength_evidence_wa_cross_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.03894723206758499,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.053662098944187164,
    "test_loss": 0.09260933101177216,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02970895729959011,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.17236286401748657,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4661589597480978,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5205428618569601,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03894723206758499,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19735053181648254,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7799096347671101,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7874950456044786,
    "epochs_trained_total": 8.626060138781805,
    "epochs_trained_best": 5.616969392695128
}
```

