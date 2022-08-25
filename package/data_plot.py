import numpy as np
import matplotlib.pyplot as plt

def compare_two_datasets(x1,y1,x2,y2,x_label,y_label,label_1,label_2,title,save_path):
    """
    input:  x1:(list) the first x value
            x2:(list) the second x value
            y1:(list) the first y value
            y2:(list) the second y value
            x_label,y_label,label_1,label_2:(str)
    """
    plt.title(title)
    plt.plot(x1, y1, color = 'blue', label = label_1)
    plt.plot(x2, y2, color='red', label = label_2)
    plt.legend() 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.show()

# Read the data
cnn_data = np.load('./data/cnn_training_data_b8.npy', allow_pickle=True).item()
unet_data = np.load('./data/unet_training_data_b8.npy', allow_pickle=True).item()

# Plot the mse loss with cnn model and unet model during training
x1 = cnn_data['step']
y1 = cnn_data['mse']
x2 = unet_data['step']
y2 = unet_data['mse']
x_label = "iterations"
y_label = "MSE Loss"
label_1 = "CNN model"
label_2 = "U-net model"
title = "The MSE loss comparison of CNN and U-net model for batch size 8"
save_path = "./data/Mse loss b8.jpg"
#compare_two_datasets(x1,y1,x2,y2,x_label,y_label,label_1,label_2,title,save_path)

# Plot the psnr with cnn model and unet model during training
y1 = cnn_data['psnr']
y2 = unet_data['psnr']
x_label = "iterations"
y_label = "psnr"
label_1 = "CNN model"
label_2 = "U-net model"
title = "The psnr comparison of CNN and U-net model for batch size 8"
save_path = "./data/psnr comparison b8.jpg"
# compare_two_datasets(x1,y1,x2,y2,x_label,y_label,label_1,label_2,title,save_path)

# Plot the ssim with cnn model and unet model during training
y1 = cnn_data['ssim']
y2 = unet_data['ssim']
x_label = "iterations"
y_label = "ssim"
label_1 = "CNN model"
label_2 = "U-net model"
title = "The ssim comparison of CNN and U-net model for batch size 8"
save_path = "./data/ssim comparison b8.jpg"
# compare_two_datasets(x1,y1,x2,y2,x_label,y_label,label_1,label_2,title,save_path)

# # plot the data of experiment
# n1 = ["30","35","40","45","50"]
# y = [0,0,0,0,0]
# i = 0
# for n in n1:
#     path = "./experiment_data/de_noise_ista_unet/noise_"+n+"/label_0/data.npy"
#     data = np.load(path,allow_pickle=True).item()
#     y[i] = data["mse"]
#     i += 1

# ex = np.load(path, allow_pickle=True).item()
# x1 = ex["iteration"]

# plt.title("MSE of ISTA method with iteration")
# plt.plot(x1, y[0], color = 'blue', label = "30% pixels missing")
# plt.plot(x1, y[1], color='red', label = "35% pixels missing")
# plt.plot(x1, y[2], color = 'yellow', label = "40% pixels missing")
# plt.plot(x1, y[3], color='green', label = "45% pixels missing")
# plt.plot(x1, y[4], color = 'black', label = "50% pixels missing")
# plt.legend() 
# plt.xlabel("Iteration")
# plt.ylabel("MSE")
# plt.savefig("./experiment_data/ista_unet.jpg")
# plt.show()