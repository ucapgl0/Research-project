import pytest
import training_data_generate
import model
from PIL import Image
import torch
import os
import torchvision.transforms as transforms

# Tests for data-generate and data-origin
path_generate = "./unit_test_data/generate-unit-test/"
path_origin = "./unit_test_data/origin-unit-test/"
path_noise = "./unit_test_data/addnoise-unit-test/"
path_output = "./unit_test_data/unit_test.txt"
path_fake = "./unit_test_data/fake_path/"
training_data_generate.generate_origin(1,path_generate,path_origin)
img = Image.open("./unit_test_data/origin-unit-test/0.jpg")
trans = transforms.ToTensor()
img_tensor = trans(img)

# test original image is grayscale (channel = 1)
def test_original_image_is_grayscale():    
    assert img_tensor.shape[0] == 1

# test the size of image is 128*128
def test_size_of_original_image():
    assert (img_tensor.shape[1],img_tensor.shape[2]) == (128,128)

# Generate noise image for unit test
training_data_generate.generate_noise(10,path_origin,path_noise,path_output)

img_noise = Image.open(path_noise+"0.jpg")

# test the output text file saving valid path
def test_valid_path_in_text():
    file = []
    output_txt = open(path_output,"r")
    for line in output_txt.readlines():
        file.append(line)
    path1 = file[0].split(" ")[0]
    path2 = file[0].split(" ")[1][:-1:]
    assert os.path.isfile(path1) and os.path.isfile(path2)

# Negative test for incorrect path
def test_improper_path_input():
    with pytest.raises(TypeError) as exception:
        training_data_generate.generate_origin(1,path_fake,path_fake)

# Test the model output same size with input image
def test_unet_model():
    input = trans(img_noise)
    cnn = model.CNN()
    output = cnn(input)
    assert input.shape == output.shape

pytest.main(["unit_test.py"])
    