import torch
from model import UNet
import random
import os
from PIL import Image
import numpy as np

def generate_noise(noise_rate,origin_path,save_path):
    """
    
    """
    # Count total number of image and use that as image filename
    original_path = os.listdir(origin_path)   
    s_matrix = []
    # random.seed(1)
    for index, filename in enumerate(original_path):         
        spare_matrix = np.zeros((128,128))   
        # Read image
        img_original = Image.open(origin_path+filename)
        # Set 25% pixel = 0 randomly 
        pixdata = img_original.load()
        coordinates = []
        x_axis = img_original.size[0]
        y_axis = img_original.size[1]
        for i in range(x_axis):
            for j in range(y_axis):                    
                coordinates.append((i,j))
        noise_num = int(x_axis*y_axis*noise_rate)
        coordinates_noise = random.sample(coordinates,noise_num)
        for coordinate in coordinates_noise:
            pixdata[coordinate[0],coordinate[1]] = 0
            if coordinate[0] == coordinate[1]:
                spare_matrix[coordinate[0],coordinate[1]] = 1
        # Save noise image in folder
        img_original.save(save_path+str(int(noise_rate*100))+"_"+str(index)+".jpg")
        s_matrix.append(spare_matrix)
    # Save spare matrix to npy file using for ista method
    np.save(save_path+str(int(noise_rate*100))+".npy",s_matrix)
            



