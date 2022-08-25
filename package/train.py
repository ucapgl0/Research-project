import numpy as np
from PIL import Image as pil
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as tt
from model import UNet, CNN
import validation
import psnr_ssim
import time
from argparse import ArgumentParser

parser = ArgumentParser(description='model-train')

parser.add_argument('--batch_size', type=int, default=8, help='number of sample in each iteration')
parser.add_argument('--model', type=int,default=1, help="pre-train model choice. 1 for Unet, 2 for CNN")
parser.add_argument('--path_train', type=str, default="./data/train.txt",help="training set")
parser.add_argument('--path_validate', type=str, default="./data/valid.txt",help="validated set")
parser.add_argument('--path_test', type=str, default="./data/test.txt",help="testing set")
parser.add_argument('--device', type=str, default="cpu", help="the device of operation")
parser.add_argument("--epoch",type=int,default=10,help="the number of training epoch")
parser.add_argument("--learning_rate",type=float,default=0.01,help="the optimizer learning rate")

args = parser.parse_args()


class mydata(Dataset):
    """
    Generate dataset for training, validation and testing
    """
    def __init__(self,path,trans):
        validation.validation_loader(path)
        self.ts = trans
        self.file = []
        f = open(path,"r")
        for line in f.readlines():
            self.file.append(line)
    def __getitem__(self,idx):       
        img = pil.open(self.file[idx].split(' ')[0])
        sign = pil.open(self.file[idx].split(' ')[1][:-1:])        
        if self.ts:
            # transform to tensor
            img = self.ts(img)
            sign = self.ts(sign)
        return {'noise':img,'origin':sign}
    def __len__(self):
        return len(self.file)
    
def train(model,train_set,valid_set,epoch,model_save,data_save):
    """
    Use for train model
    input:
        model: neural network
        train_set: add-noise image and original image used for train
        valid_set: used for model validation
        epoch: number of training epoch
        model_save: the path of the best model
        data_save: the path of training data save
    output:
        model: trained model
    """
    LR = args.learning_rate
    # Choose Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    # Set loss function
    loss_func = nn.MSELoss()
    global_step = 0
    # Set three lists of step, loss respctivly used for plot
    step_data = []
    loss_data = []
    psnr_value = []
    ssim_value = []
    if BATCHSIZE == 1:
        valid_iter = 100
    else:
        valid_iter = 25
    min_loss = np.inf
    for i in range(epoch):
        for index, data in enumerate(train_set):
            x_in = data['noise'].cpu()
            y_in = data['origin'].cpu()
            model.train()
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                x_out = model(x_in)
                # Calculate loss
                loss = loss_func(x_out, y_in)        
                loss.backward(retain_graph=True)
                optimizer.step()
                global_step += 1
            # validation every 100 iterations 
            if global_step % valid_iter == valid_iter - 1:
                valid_loss,valid_psnr,valid_ssim = test(model,valid_set)
                # ssim = []
                step_data.append(global_step+1)
                loss_data.append(loss.item())
                # psnr_value.append(psnr_ssim.psnr_calculation(loss.item()))
                psnr_value.append(valid_psnr)
                ssim_value.append(valid_ssim)
                
                if valid_loss < min_loss:
                    min_loss = valid_loss
                    # Save the best model
                    torch.save(model.state_dict(), model_save) # save best model weight
                    break 
    training_data = {"step":step_data, "mse":loss_data, "psnr":psnr_value, "ssim":ssim_value}
    # np.save(data_save,training_data)
    return model

def test(model,data_loader,save_pic=False,save_path=None):
    """
    Use for test model
    input:
        model: neural network model for validating or testing
        data_loader: torch data_loader used for validated or testing
        save_pic: (bool) whether save image or not
        save_path: the path for image saving
    output:
        loss: MSE, PSNR, SSIM of tested images with original images
    """
    model.eval() 
    mse_loss = nn.MSELoss()
    toPIL = tt.ToPILImage()
    valid_mse_loss = []
    psnr_value = []
    ssim_value = []
    for index, data in enumerate(data_loader):
        test_x = data['noise'].cpu()
        test_y = data['origin'].cpu()
        ssim = []
        # We don't need to compute the gradient, so we don't have to differentiate
        with torch.no_grad():
            test_out = model(test_x)
            mse = mse_loss(test_out,test_y).item()
            valid_mse_loss.append(mse)
            psnr_value.append(psnr_ssim.psnr_calculation(mse))
            # Transform from [b,c,h,w] to [h,w] for ssim calculation
            img_noise = torch.squeeze(test_out)
            img_origin = torch.squeeze(test_y)
            if BATCHSIZE == 1:
                    ssim_value.append(psnr_ssim.ssim_calculation(img_noise.detach().numpy()*255,img_origin.detach().numpy()*255))
            else:
                for j in range(img_noise.shape[0]):
                    ssim.append(psnr_ssim.ssim_calculation(img_noise[j].detach().numpy()*255,img_origin[j].detach().numpy()*255))
                ssim_value.append(np.mean(ssim))
        if save_pic == True:
            if BATCHSIZE > 1:
                for i in range(BATCHSIZE):                
                    pic = toPIL(test_out[i])
                    pic.save(save_path+str(2*index+i)+".jpg")
            else:
                pic = toPIL(test_out[0])
                pic.save(save_path+str(index)+".jpg")    
    return np.mean(valid_mse_loss), np.mean(psnr_value), np.mean(ssim_value) 

if __name__ == '__main__':
    BATCHSIZE = args.batch_size
    # Use cpu as device, since gpu is out of memory
    device = torch.device(args.device)
    path_train = args.path_train
    path_valid = args.path_validate
    path_test = args.path_test
    trans = tt.Compose([tt.ToTensor()])

    dataset_train = mydata(path_train,trans)
    train_loader = DataLoader(dataset_train,batch_size=BATCHSIZE,num_workers=0,shuffle=True)

    dataset_valid = mydata(path_valid,trans)
    valid_loader = DataLoader(dataset_valid,batch_size=BATCHSIZE,num_workers=0,shuffle=True)

    dataset_test = mydata(path_test,trans)
    test_loader = DataLoader(dataset_test,batch_size=BATCHSIZE,num_workers=1)

    epoch = args.epoch
    
    if args.model == 1:
        net = UNet().to(device)
        model_name = "best_unet_b"+str(BATCHSIZE)+".mdl"
        data_save_path = "./data/unet_training_data_b"+str(BATCHSIZE)+".npy"
        test_img_save = "./data/de-noise-unet-b"+str(BATCHSIZE)+"/"
        
    elif args.model == 2:
        net = CNN().to(device)
        model_name = "best_cnn_b"+str(BATCHSIZE)+".mdl"
        data_save_path = "./data/cnn_training_data_b"+str(BATCHSIZE)+".npy"
        test_img_save = "./data/de-noise-cnn-b"+str(BATCHSIZE)+"/"
    

    
    start = time.time()
    model = train(net,train_loader,valid_loader,epoch,model_name,data_save_path)
    end = time.time()
    print("training time: " + str(end-start))
    # model.load_state_dict(torch.load('best_unet_model.mdl'))
    # Test model  
    model.load_state_dict(torch.load(model_name))   
    test_loss,test_psnr,test_ssim = test(model,test_loader,True,test_img_save)
    print({"mse":test_loss,"psnr":test_psnr,"ssim":test_ssim})

    
    

    
