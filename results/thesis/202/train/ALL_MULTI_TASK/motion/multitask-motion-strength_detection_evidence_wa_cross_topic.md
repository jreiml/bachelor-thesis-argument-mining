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
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5309992432594299,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5320171117782593,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.38836371898651123,
    "test_loss": 1.4513800740242004,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.029241029173135757,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.17100007832050323,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4680307954207805,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5260019158475846,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7605092966855296,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7460933803953229,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7716483125806763,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7605092966855295,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7498030386450631,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8332958718710879,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7605092966855295,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8031584414985251,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7605092966855295,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8065937653011261,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.6855929954895198,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9474693251533742,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.5521367521367522,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7021881216254617,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.9041287613715885,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03883637487888336,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.1970694661140442,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7843127159093206,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7940142868166746,
    "epochs_trained_total": 6.8585365853658535,
    "epochs_trained_best": 3.8327116212338592
}
```

