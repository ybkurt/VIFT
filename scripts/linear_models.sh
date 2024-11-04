python src/train.py experiment=latent_kitti_vio trainer=gpu logger=tensorboard model.net.lin1_size=256 \
    model.net.lin2_size=128 model.net.lin3_size=64

python src/train.py experiment=latent_kitti_vio trainer=gpu logger=tensorboard model.net.lin1_size=512 \
    model.net.lin2_size=512 model.net.lin3_size=128