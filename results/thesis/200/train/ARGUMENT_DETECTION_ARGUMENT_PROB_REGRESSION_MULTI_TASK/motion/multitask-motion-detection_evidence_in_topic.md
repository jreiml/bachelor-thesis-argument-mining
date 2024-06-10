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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4012829065322876,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.30667468905448914,
    "test_loss": 0.7079575955867767,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8171853591700919,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.816714320832117,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8165706450456306,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8171853591700919,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8224922813611653,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8231945373964998,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8171853591700919,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8181648735191013,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8171853591700919,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.826005961251863,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8074226804123711,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7770767613038907,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8679078014184397,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8815109343936381,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7548188126445644,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03319280222058296,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.18218891322612762,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8101013951840433,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8314069560338163,
    "epochs_trained_total": 4.707136997538966,
    "epochs_trained_best": 1.7025814671949453
}
```

