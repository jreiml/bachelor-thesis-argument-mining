# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 14.8 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_evidence_wa_in_topic \
    --run_name multitask-no_motion-strength_detection_evidence_wa_in_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4912065863609314,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.3841504156589508,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4147094488143921,
    "test_loss": 1.2900664508342743,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02596367709338665,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16113248467445374,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.49198899585784606,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5401748669941352,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7956547269524369,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7956033495661292,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7953274339729948,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7956547269524369,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8047911492941361,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8114483414526605,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7956547269524369,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8028246892175005,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7956547269524369,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7988439306358383,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7923627684964201,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7266035751840169,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8829787234042553,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8870346598202824,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7186147186147186,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03949613869190216,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19873635470867157,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.780995999751048,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8021466293370834,
    "epochs_trained_total": 5.9375,
    "epochs_trained_best": 2.9184322033898304
}
```

