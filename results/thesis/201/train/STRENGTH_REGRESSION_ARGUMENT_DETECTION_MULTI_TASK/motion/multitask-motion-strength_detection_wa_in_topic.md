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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.37975573539733887,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4226120114326477,
    "test_loss": 0.8023677468299866,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025652289390563965,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16016332805156708,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.510408985532301,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5538195186373082,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8185554903112156,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.817993080648586,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8179356514928949,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8185554903112156,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8233016664056498,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8235322176732187,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8185554903112156,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8188974079193779,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8185554903112156,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.828110513628778,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8078756476683939,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7826848930949877,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8639184397163121,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8791338582677165,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7586609575710394,
    "epochs_trained_total": 5.984714400643604,
    "epochs_trained_best": 2.9416392816722796
}
```

