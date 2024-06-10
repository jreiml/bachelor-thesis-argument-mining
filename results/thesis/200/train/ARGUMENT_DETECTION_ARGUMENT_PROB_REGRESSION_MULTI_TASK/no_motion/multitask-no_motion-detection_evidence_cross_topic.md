# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 7.6 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_cross_topic \
    --run_name multitask-no_motion-detection_evidence_cross_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4768737852573395,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4066549241542816,
    "test_loss": 0.8835287094116211,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.769199676637025,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7596557250060271,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7761495318815043,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.769199676637025,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7606378795029102,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8150755747138496,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.769199676637025,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7948410053288628,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.769199676637025,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8075497135153353,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7117617364967188,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9187116564417178,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.6025641025641025,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7203848466626579,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8692971639950678,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.053507234901189804,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.23131631314754486,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.6920887533688524,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.696209015834285,
    "epochs_trained_total": 4.141295206055509,
    "epochs_trained_best": 1.1110792016246487
}
```

