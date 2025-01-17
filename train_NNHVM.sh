nohup python train.py --epochs 300 \
                      --batch_size 256 \
                      --lr 0.0002 \
                      --latent_dim 100 \
                      --img_size 32 \
                      --channels 3 \
                      --sample_interval 50 \
                      --b1 0.5 \
                      --b2 0.999 \
                      --k 100 \
                      --model NNHVM \
                      --save_model_path /home/wangchai/zhw/Generative-Model/ckpt/NNHVM.pth \
                      --dataset_path /home/wangchai/zhw/Generative-Model/dataset > train_NNHVM.log 2>&1 &