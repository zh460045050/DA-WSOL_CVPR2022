data_root=""
CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name CUB \
        --data_root $data_root \
        --architecture resnet50 \
        --wsol_method cam \
        --uda_method mmd \
        --experiment_name MMD_CAM_RES_CUB \
        --pretrained TRUE \
        --num_val_sample_per_class 0 \
        --large_feature_map TRUE \
        --batch_size 32 \
        --epochs 50 \
        --lr 1E-3 \
        --lr_decay_frequency 30 \
        --weight_decay 1.00E-04 \
        --override_cache FALSE \
        --workers 16 \
        --box_v2_metric True \
        --iou_threshold_list 30 50 70 \
        --eval_checkpoint_type last \
        --has_grid_size 50 \
        --has_drop_rate 0.66 \
        --save_dir 'train_TL_search_CUB_MUD' \
        --seed 4 \
        --check_path "" \
        --beta 0.3 \
        --univer 2 \
        --eval_frequency 5
