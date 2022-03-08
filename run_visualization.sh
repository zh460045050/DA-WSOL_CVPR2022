data_root=""

CUDA_VISIBLE_DEVICES=1 python visualization.py \
                --dataset_name OpenImages \
                --split test \
                --iou_threshold_list 30 50 70 \
                --cam_curve_interval 0.01 \
                --scoremap_root "test_logs/OPEN_MMD_CAM_RES/scoremaps" \
                --mask_root $data_root \
                --img_root $data_root \
                --save_dir "visual_res"\
                --check_name "DA-WSOL"
