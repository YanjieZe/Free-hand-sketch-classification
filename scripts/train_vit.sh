gpu_used=1

CUDA_VISIBLE_DEVICES=$gpu_used python src/main.py \
    --use_gpu \
    --alg vit \
    --data_dir dataset/quickdraw_png  \
    --batch_size 128