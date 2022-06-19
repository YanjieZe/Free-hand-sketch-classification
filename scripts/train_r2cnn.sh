gpu_used=1

CUDA_VISIBLE_DEVICES=$gpu_used python src/main.py \
    --use_gpu \
    --alg sketch_r2cnn \
    --img_form svg \
    --data_dir dataset/quickdraw_svg \
    --batch_size 128 \