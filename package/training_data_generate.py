import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import numpy as np
import random
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import validation
from random import sample
from argparse import ArgumentParser

parser = ArgumentParser(description='data-generate')

parser.add_argument('--origin_num', type=int, default=10, help='number of number of original images')
parser.add_argument('--generate_path', type=str, default="./data/generate/", help="the path save generated RGB images")
parser.add_argument('--origin_path', type=str, default="./data/origin/",help="the path save original images")
parser.add_argument('--train_path', type=str, default="./data/train/",help="the path save training images")
parser.add_argument('--train_num', type=int, default=100,help="the number of training images each label")
parser.add_argument('--train_output', type=str,default="./data/train.txt",help="the output training txt file path")
parser.add_argument('--validate_path', type=str, default="./data/valid/",help="the path save validated images")
parser.add_argument('--validate_num', type=int, default=10,help="the number of validated images each label")
parser.add_argument('--validate_output', type=str,default="./data/valid.txt",help="the output validated txt file path")
parser.add_argument('--test_path', type=str, default="./data/test/",help="the path save testing images")
parser.add_argument('--test_num', type=int, default=20,help="the number of testing images each label")
parser.add_argument('--test_output', type=str,default="./data/test.txt",help="the output testing txt file path")

args = parser.parse_args()


def generate_origin(num,path,path_origin):
    """
    Generate original elipses image and save them in spetific folder
    input: 
        num: the number of ellipse
        path: spetific path
    return:
        None
    """
    validation.validation_generate(num,path,path_origin)
    for i in range(num):    
        NUM = 50
        ells = [Ellipse(xy=np.random.rand(2) * 120,
            width=np.random.rand()*100, height=np.random.rand()*100,
            angle=np.random.rand() * 360)
            for j in range(NUM)]
 
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        origin_transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                                                transforms.Grayscale(),transforms.ToPILImage()])
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(np.random.rand())
            e.set_facecolor(np.random.rand(3))
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        plt.axis("off")      
        # save in folder and the last element of file name is the label of image
        plt.savefig(path + "ellipse_"+str(i)+".jpg",bbox_inches='tight',pad_inches=0.0)
        
        img_original = Image.open(path + "ellipse_"+str(i)+".jpg")            
        img_origin = origin_transform(img_original)

        img_origin.save(path_origin+str(i)+".jpg")


    
def generate_noise(size,path_origin,path_save,output_txt,generate_test=False):                   
    """                  
    Set 25% pixel equal to 0 randomly of all original image to generate training(testing) set and save in spetific
    folder and write data in .txt file
    input:
        size: size of each class of training(testing) set (the number of image which is same class(label) as an original image)
        path_input: the folder saving original image
        path_save: the folder saving generated training image
        output_txt: the .txt file saving training image and add-noise image in every line
    output:
        None
    """
    validation.validation_generate(size,path_origin,path_save,output_txt)
    # Count total number of image and use that as image filename
    count = 0
    original_path = os.listdir(path_origin)
    data_list = [0 for i in range(size*len(original_path))]    
    
    for filename in original_path:
        random.seed(1)
        for i in range(size):
            # Read image
            img_original = Image.open(path_origin+filename)
            # Set 25% pixel = 0 randomly 
            pixdata = img_original.load()
            coordinates = []
            x_axis = img_original.size[0]
            y_axis = img_original.size[1]
            for i in range(x_axis):
                for j in range(y_axis):                    
                    coordinates.append((i,j))
            noise_num = int(x_axis*y_axis*0.25)
            coordinates_noise = sample(coordinates,noise_num)
            for coordinate in coordinates_noise:
                pixdata[coordinate[0],coordinate[1]] = 0
            # save image in folder
            img_original.save(path_save+str(count)+".jpg")
            # append the path of noise and origin of image to list
            data_list[count] = path_save+str(count)+".jpg"+" "+path_origin+filename
            count += 1
    # random order the datalist and write it to .txt file, which is helpful to train
    if generate_test == False:
        random.shuffle(data_list)
    with open(output_txt, 'a+') as f:
        for line in data_list:
            f.write(line+'\n')
    
path_generate = args.generate_path

path_original = args.origin_path
origin_num = args.origin_num

path_train = args.train_path
train_txt = args.train_output
train_num = args.train_num

path_valid = args.validate_path
valid_txt = args.validate_output
valid_num = args.validate_num

path_test = args.test_path
test_txt = args.test_output
test_num = args.test_num

if __name__ == "__main__":
    pass
    # # Generate 10 original elipses images
    # generate_origin(origin_num, path_generate, path_original)

    # # Generate training elipses images
    # generate_noise(train_num,path_original,path_train,train_txt)

    # # Generate valid elipses images
    # generate_noise(valid_num,path_original,path_valid,valid_txt)

    # # Generate test elipses images
    # generate_noise(test_num,path_original,path_test,test_txt,True)

