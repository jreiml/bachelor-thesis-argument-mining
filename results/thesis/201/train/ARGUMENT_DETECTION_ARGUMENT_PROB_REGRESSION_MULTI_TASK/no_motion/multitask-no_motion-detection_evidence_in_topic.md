# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_in_topic \
    --run_name multitask-no_motion-detection_evidence_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.41582930088043213,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.45165953040122986,
    "test_loss": 0.867488831281662,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8017224505774123,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8013739592882048,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8010987488775688,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8017224505774124,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8078587955194607,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8095958847911627,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8017224505774124,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8039877545322192,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8017224505774124,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8096937817020479,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7930541368743617,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7553452506133894,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8603723404255319,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8724696356275303,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7355058734369079,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03960279002785683,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19900450110435486,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7833936123979949,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.803826066033978,
    "epochs_trained_total": 6.009844134536506,
    "epochs_trained_best": 3.004922067268253
}
```

