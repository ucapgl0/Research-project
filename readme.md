# Study of Generalisation Scenarios for CNNs on an inpainting problem

## 1. introduction

The purpose of this project is to solve a greyscale image inpainting problem by training neural network models and implementing ISTA algorithm. The functions of the program include image data generation, model training, validation and testing, and plot the experiment data for model effect analysis.

## 2. package installation instruction

Firstly, open the timinal and input

```Bash
git clone https://github.com/ucapgl0/Research-project
```
The code will be cloned in current path
and then type in
```Bash
cd Research-project
```
finally, install the package by inputing
```Bash
pip install .
```

## 3, Instruction of function usage

After installing package, type
```Bash
cd package
```
and user could generate greyscale image data for training, validation and testing by input 
```Bash
python training_data_generate.py --origin_num<default=10> --generate_path<default="./data/generate/">
--origin_path<default="./data/origin/">
--train_path<default="./data/train/">
--train_num<default=100>
--train_output<default="./data/train.txt">
--validate_path<default="./data/valid/">
--validate_num<default=10>
--validate_output<default="./data/valid.txt">
--test_path<default="./data/test/">
--test_num<default=20>
--test_output<default="./data/test.txt">
```

The data has been generated and user could skip this step and train and test the models using existing data and type
```Bash
python train.py --batch_size<default=8>
--model<default=1,help="pre-train model choice. 1 for Unet, 2 for CNN">
--path_train<default="./data/train.txt">
--path_validate<default="./data/valid.txt">
--path_test<default="./data/test.txt">
--device<default="cpu">
--epoch<default=10>
--learning_rate<default=0.01>
```

Also user can do experiment of ISTA by typing
```Bash
python --num_image<default=5>
--noise_data<default=[0.3,0.35,0.4,0.45,0.5]>
```

finally, user can do unit test by input
```Bash
python unit_test.py
```