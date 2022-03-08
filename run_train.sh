data_root="/mnt/nas/zhulei/datas/dataset/"
###Training our DA-WSOL pipeline
CUDA_VISIBLE_DEVICES=1 python main.py --data_root $data_root \
                --experiment_name OPEN_MMD_CAM_RES \
                --pretrained TRUE \
                --num_val_sample_per_class 0 \
                --large_feature_map False \
                --batch_size 32 \
                --epochs 10 \
                --lr 1E-3 \
                --lr_decay_frequency 3 \
                --weight_decay 1.00E-04 \
                --override_cache FALSE \
                --workers 16 \
                --box_v2_metric True \
                --iou_threshold_list 30 50 70 \
                --save_dir 'train_logs' \
                --seed 4 \
                --dataset_name OpenImages \
                --architecture resnet50 \
                --wsol_method cam \
                --uda_method mmd \
                --beta 0.2 \
                --univer 3 \
                --start_epoch 0 \
                --check_path "" \
                --eval_frequency 1