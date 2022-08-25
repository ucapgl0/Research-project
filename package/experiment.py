import experiment_data_generate
import training_data_generate
from model import UNet
import ista
import torch
import numpy as np
import torchvision.transforms as tt
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser(description='model-train')

parser.add_argument('--num_image', type=int, default=5, help='number of images for experiment')
parser.add_argument('--noise_data',type=list, default=[0.3,0.35,0.4,0.45,0.5],help="noise rate list of images")

args = parser.parse_args()
origin_path = "./experiment_data/origin/"
noise_path = "./experiment_data/add_noise/"
# training_data_generate.generate_origin(5,generate_path,origin_path)
noise_data = args.noise_data
for noise_rate in noise_data:
    experiment_data_generate.generate_noise(noise_rate, origin_path, noise_path)
num_label = args.num_image
label = ["0","1","2","3","4"]
# load the model
net = UNet().to(device="cpu")
net.load_state_dict(torch.load('best_unet.mdl'))
for n in noise_data:
    for l in range(num_label):
        # load diagonal matrix
        spare_matrix = np.load(noise_path+str(int(n*100))+".npy")
        save_path = "./experiment_data/de_noise_ista_unet/noise_"+str(int(n*100))+"/label_"+str(l)+"/"
        # read obscured and original image
        img_noise = Image.open(noise_path+str(int(n*100))+"_"+str(l)+".jpg")
        img_origin = Image.open(origin_path+str(l)+".jpg")
        trans = tt.ToTensor()
        ista.ista_method(trans(img_noise),trans(img_origin),net,spare_matrix[int(l)],50,save_path)
