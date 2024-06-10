# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK \
    --task_head_loss_weights 14.8 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_wa_in_topic \
    --run_name multitask-motion-strength_detection_wa_in_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.43861132860183716,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.3787470757961273,
    "test_loss": 0.8173584043979645,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025591017678380013,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1599719226360321,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.49997181540020375,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5447718904184703,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8130749657467214,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8127525503093476,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8124894471458302,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8130749657467214,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8194148470068834,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8212520320286724,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8130749657467214,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8154150483247284,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8130749657467214,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8205224581845518,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8049826424341433,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7651594812478093,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8736702127659575,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8845218800648298,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.746308216584627,
    "epochs_trained_total": 5.477876106194691,
    "epochs_trained_best": 2.4346116027531957
}
```

