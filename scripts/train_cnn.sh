gpu_used=1

CUDA_VISIBLE_DEVICES=$gpu_used python src/main.py \
    --use_gpu \
    --alg cnn \
    --patch_size 7 \
    --data_dir dataset/quickdraw_png