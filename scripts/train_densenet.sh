gpu_used=2

CUDA_VISIBLE_DEVICES=$gpu_used python src/main.py \
    --use_gpu \
    --alg densenet \
    --data_dir dataset/quickdraw_png \
    --batch_size 64 \