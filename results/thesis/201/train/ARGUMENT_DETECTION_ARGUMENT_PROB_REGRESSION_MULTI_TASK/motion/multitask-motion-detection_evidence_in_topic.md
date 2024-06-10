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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.41173624992370605,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.31190523505210876,
    "test_loss": 0.7236414849758148,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8128792327265609,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8125472642101201,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.812289409595024,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8128792327265609,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8191468387140077,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8209000886300941,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8128792327265609,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8151257390889741,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8128792327265609,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8204357625845229,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8046587658357173,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7655099894847529,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8727836879432624,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8838526912181303,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7463987869598181,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.033748943358659744,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.18370886147022247,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8109907705732906,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8283809289826756,
    "epochs_trained_total": 4.907301066447908,
    "epochs_trained_best": 1.9028310257655152
}
```

