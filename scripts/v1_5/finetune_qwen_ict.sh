#!/bin/bash

ds_array=("gqa" "ocr_vqa" "basic_qa_geo170k" "textvqa" "vg" "conv" "mavis_math_metagen" "det" "tabmwp_cauldron" "complex_res" "refcoco" "reasoning_qa_geo170k" "okvqa" "vqav2" "aokvqa")
ds_array2=("")

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --version qwen_2 \
    --data_path datasets/LLaVA-Instruct \
    --k_spt 10 \
    --data_mode "ict" \
    --mix_no_shot True \
    --image_folder Image_data \
    --train_datasets "${ds_array[@]}" \
    --val_datasets "${ds_array2[@]}" \
    --remove_instruct True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter pretrained_checkpoint/mm_projector.bin \
    --mm_projector_type attention_mapper \
    --prefix_length 256 \
    --add_dropout False \
    --extrapolate_lr False \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir llava_weights/checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 5  \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "ict_trial1" \
    --project_name "MAPD_train"