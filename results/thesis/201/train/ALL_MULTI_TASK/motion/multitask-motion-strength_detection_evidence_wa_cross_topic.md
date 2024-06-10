# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 18.2 1 10 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_evidence_wa_cross_topic \
    --run_name multitask-motion-strength_detection_evidence_wa_cross_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4298701584339142,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5107775926589966,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.40112125873565674,
    "test_loss": 1.3417690098285675,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.028064701706171036,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16752523183822632,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.49174161262044974,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5427154001814933,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8039611964430072,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7997923322443818,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8065652768808041,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8039611964430072,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7984878873682555,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8221783296321066,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8039611964430072,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8149206573130328,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8039611964430072,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8286824443659483,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7709022201228153,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.8995398773006135,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.6974358974358974,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.768172888015717,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8616684266103485,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.04011688753962517,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.20029200613498688,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7840809876667012,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7893416217368282,
    "epochs_trained_total": 4.639566395663957,
    "epochs_trained_best": 1.6137622245787677
}
```

