#Runing Demo with our checkpoint
check_path="checkpoints/dawsol_openimages.tar"

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_imgs" \
                --check_path $check_path \
                --save_path "./demo_results" \
                #--img_name "burger.jpg"