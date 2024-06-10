# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 18.2 1 7.6 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_evidence_wa_cross_topic \
    --run_name multitask-no_motion-strength_detection_evidence_wa_cross_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4557443857192993,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5224708914756775,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.39316239953041077,
    "test_loss": 1.3713776767253876,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.028707189485430717,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16943195462226868,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4777000697502622,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.527548385308564,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7886014551333871,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7823894306580341,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7928220650640098,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7886014551333872,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7817435504168633,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8176707825885036,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7886014551333872,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.805515477253214,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7886014551333872,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8191562932226832,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7456225680933851,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9083588957055214,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.6551282051282051,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7459068010075567,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8651241534988713,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05173025652766228,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.22744286060333252,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7114956739304189,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7121876788458288,
    "epochs_trained_total": 4.639566395663957,
    "epochs_trained_best": 1.6137622245787677
}
```

