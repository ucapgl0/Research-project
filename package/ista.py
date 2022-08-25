import psnr_ssim
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as tt

def ista_method(noise_image,origin_image,proximal_operator,spare_matrix,iterations,data_save_path):
    proximal_operator.eval()
    mse_loss = nn.MSELoss()
    origin_image = torch.unsqueeze(origin_image,0)
    origin_np = torch.squeeze(origin_image).detach().numpy()
    y = noise_image
    y_array = torch.squeeze(y).numpy()
    x_i = noise_image
    
    trans_tensor = tt.ToTensor()
    trans_img = tt.ToPILImage()
    step = []
    loss_data = []
    psnr_value = []
    ssim_value = []   
    for i in range(iterations):        
        x_i_np = torch.squeeze(x_i).detach().numpy()
        tilde_x = x_i_np - 0.1*spare_matrix.transpose()*(spare_matrix*x_i_np - y_array)
        tilde_x = torch.unsqueeze(trans_tensor(tilde_x), 0)
        x_i = proximal_operator(tilde_x.float())              
        x_array = torch.squeeze(x_i).detach().numpy()
        mse = float(mse_loss(x_i,origin_image))
        step.append(i+1)
        loss_data.append(mse)
        psnr_value.append(psnr_ssim.psnr_calculation(mse))
        ssim_value.append(psnr_ssim.ssim_calculation(x_array*255,origin_np*255))
        if i >= 0:
            iter_img = trans_img(torch.squeeze(x_i,0))
            iter_img.save(data_save_path+str(i)+".jpg")
    experiment_data = {"iteration":step, "mse":loss_data, "psnr":psnr_value, "ssim":ssim_value}
    np.save(data_save_path+"data.npy",experiment_data)

