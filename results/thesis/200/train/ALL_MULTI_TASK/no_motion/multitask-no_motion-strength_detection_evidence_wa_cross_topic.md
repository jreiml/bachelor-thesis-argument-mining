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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.39968523383140564,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5349230766296387,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5228070020675659,
    "test_loss": 1.4574153125286102,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.0294080451130867,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1714877337217331,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.46541462361242814,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5188124439133373,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7489894907033144,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7349704109858434,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7597070648282706,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7489894907033144,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7386994651565204,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8157978655220524,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7489894907033144,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.784923308023612,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7489894907033144,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7959250739401906,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.6740157480314961,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.928680981595092,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.5487179487179488,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6963772282921219,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8734693877551021,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05235559493303299,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.22881345450878143,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.6999631249183444,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7057609406032498,
    "epochs_trained_total": 5.547425474254743,
    "epochs_trained_best": 2.521557033752156
}
```

