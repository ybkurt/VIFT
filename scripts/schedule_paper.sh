#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh


python src/train.py experiment=latent_kitti_vio_weighted trainer=gpu logger=tensorboard  tags=['MLP, 2, l1, 10, no, no']
python src/train.py experiment=latent_kitti_vio_paper_mse trainer=gpu logger=tensorboard tags=['TE, 11, L2, 100, no, no']
python src/train.py experiment=latent_kitti_vio_paper model.criterion.angle_weight=1 trainer=gpu logger=tensorboard tags=['TE, 11, L1, 1, no, no']
python src/train.py experiment=latent_kitti_vio_paper model.criterion.angle_weight=10 trainer=gpu logger=tensorboard tags=['TE, 11, L1, 10, no, no']
python src/train.py experiment=latent_kitti_vio_paper model.criterion.angle_weight=40 trainer=gpu logger=tensorboard tags=['TE, 11, L1, 40, no, no']
python src/train.py experiment=latent_kitti_vio_paper model.criterion.angle_weight=100 trainer=gpu logger=tensorboard tags=['TE, 11, L1, 100, no, no']
python src/train.py experiment=data_l1 trainer=gpu logger=tensorboard tags=['TE, 11, L1, 10, yes, no']
python src/train.py experiment=latent_kitti_vio_weighted_tf trainer=gpu logger=tensorboard  tags=['TE, 11, L1, 40, no, yes']
python src/train.py experiment=latent_kitti_vio_weighted_tf model.net.dim_feedforward=256 trainer=gpu logger=tensorboard  tags=['feedforward_256']
python src/train.py experiment=latent_kitti_vio_weighted_tf model.net.dim_feedforward=64 trainer=gpu logger=tensorboard  tags=['feedforward_64']
python src/train.py experiment=data_rpmg trainer=gpu logger=tensorboard tags=['TE, 11, L1, 40, yes, yes']
python src/train.py experiment=latent_kitti_vio_weighted_tf_64 trainer=gpu logger=tensorboard  tags=['TE, 65, L1, 40, yes, yes']

