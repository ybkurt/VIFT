# we will use many functions inside the parent folder
import sys
sys.path.insert(0, '../')
# load the dataloader and define transforms
from src.data.components.KITTI_dataset import KITTI
import torch
from src.utils import custom_transform
# you could utilize augmentations here
transform_train = [custom_transform.ToTensor(),
                   custom_transform.Resize((256, 512))]
transform_train = custom_transform.Compose(transform_train)
dataset = KITTI("kitti_data", train_seqs=['00','01','02','04','06', '08', '09'], transform=transform_train, sequence_length=11)
save_dir = "kitti_latent_data/train_10"
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# we need to define helpers to convert the dictionary to object
class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
# Define the FeatureEncoder
from src.models.components.vsvio import Encoder
params = {"img_w": 512, "img_h": 256, "v_f_len": 512, "i_f_len": 256 ,"imu_dropout": 0.1, "seq_len":11}
params = ObjFromDict(params)

# we will define a wrapper model for feature encoder to load the pretrained feature_net weights
class FeatureEncodingModel(torch.nn.Module):
    def __init__(self, params):
        super(FeatureEncodingModel, self).__init__()
        self.Feature_net = Encoder(params)
    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i
    
model = FeatureEncodingModel(params)
# load the pretrained weights
# Load the weights to encoder from pretrained model
pretrained_w = torch.load("../pretrained_models/vf_512_if_256_3e-05.model", map_location='cpu')
model_dict = model.state_dict()
update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
# update the model dict

#check if all parameters are loaded
assert len(update_dict) == len(model_dict), "Some weights are not loaded"

model_dict.update(update_dict)
model.load_state_dict(model_dict)
# freeze the weights
for param in model.Feature_net.parameters():
    param.requires_grad = False

# create a directory to save the latent vectors
import os

os.makedirs(save_dir, exist_ok=True)

# loop over the dataset, save the latent vectors by concatenating the feature vectors
import numpy as np
model.eval()
model.to("cuda")
# use tqdm
from tqdm import tqdm
with torch.no_grad():
    for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
        imgs = imgs.to("cuda").float()
        imus = imus.to("cuda").float()
        feat_v, feat_i = model(imgs, imus)
        latent_vector = torch.cat((feat_v, feat_i), 2)
        latent_vector = latent_vector.squeeze(0)
        np.save(os.path.join(save_dir,f"{i}.npy"), latent_vector.cpu().detach().numpy())
        # also save the ground truth, rotation and weight
        np.save(os.path.join(save_dir,f"{i}_gt.npy"), gts.cpu().detach().numpy())
        np.save(os.path.join(save_dir,f"{i}_rot.npy"), rot.cpu().detach().numpy())
        np.save(os.path.join(save_dir,f"{i}_w.npy"), w.cpu().detach().numpy())
