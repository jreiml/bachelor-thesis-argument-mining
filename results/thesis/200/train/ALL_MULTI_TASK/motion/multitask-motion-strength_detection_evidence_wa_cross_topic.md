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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4016348719596863,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.37858667969703674,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5067446827888489,
    "test_loss": 1.286966234445572,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.027843112125992775,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16686254739761353,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4918842084006441,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5457399941846067,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8231608730800323,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8214453552057928,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8239284346856199,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8231608730800324,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8202581799591002,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8278615029848254,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8231608730800324,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8258781403840783,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8231608730800324,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8389471746732928,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8039435357382928,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.8738496932515337,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.7666666666666667,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8067256637168142,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8450306170513424,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03831394389271736,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.1957394778728485,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7836565534669887,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7902385177442769,
    "epochs_trained_total": 5.849864498644987,
    "epochs_trained_best": 2.824072516587235
}
```

