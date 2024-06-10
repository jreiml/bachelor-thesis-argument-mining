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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.035989910364151,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.03995927795767784,
    "test_loss": 0.07594918832182884,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025603624060750008,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16001132130622864,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4994888337380265,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.55082850860284,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03995928168296814,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19989818334579468,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7826835685405861,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8039227677900528,
    "epochs_trained_total": 7.634703196347032,
    "epochs_trained_best": 4.621004566210046
}
```

