#!/bin/bash

train_ds=("blip")
val_ds=("")

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --version plain \
    --data_path datasets/LLaVA-Pretrain \
    --data_mode "supervised" \
    --image_folder Image_data \
    --train_datasets "${train_ds[@]}" \
    --val_datasets "${val_ds[@]}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type attention_mapper \
    --prefix_length 256 \
    --add_dropout False \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir llava_weights/checkpoints \
    --num_train_epochs 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --prediction_loss_only True \
    --evaluation_strategy "no" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --metric_for_best_model "eval_loss" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "pre_trial1" \
    --project_name "MAPD_train" \
