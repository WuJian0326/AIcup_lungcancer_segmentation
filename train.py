import cv2

from models.model import ResNet
# from models.Unet import ResNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms , models
from trainer import trainer
from DataLoader import *
from utils_tool import *
from loss_fn import *
# import onnx
from semseg.models.segformer import SegFormer

lr = 5e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 30
num_epoch = 500
num_worker = 4
pin_memory = True
resume = False
train_image_path = 'SEG_Train_Datasets/Mix_858_471/Train_Images/'
train_mask_path = 'SEG_Train_Datasets/Mix_858_471/Train_Mask/'
#'SEG_Train_Datasets/Org_Image/Train_Images/'
#'SEG_Train_Datasets/Org_Image/Train_Mask/'
def main():
    train_transform = get_train_transform() #取得影像增強方式
    vaild_transform = get_vaild_transform() #取得測試影像增強
    #將資料送進dataloader中
    train_data = ImageMaskDataset(train_image_path, train_mask_path, train_transform) #
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    #建立模型
    model = ResNet().to(device)
    model = nn.DataParallel(model)
    # model = load_checkpoint(model)
    loss_function = Subloss()  #Subloss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96)

    train = trainer(train_loader, model, optimizer, scheduler, loss_function, epochs=num_epoch,best_acc=None)
    #訓練
    model = train.training()

if __name__ == '__main__':
    #訓練
    main()
    #預測
    model = ResNet().to(device)
    model2 = ResNet().to(device)
    model = load_checkpoint(model, path='checkpoint/random_858_471/ckpt_0.pth')
    model2 = load_checkpoint(model,path='checkpoint/ckpt_Mix.pth')
    model2.eval()
    model.eval()


    #test=================================================
    dummy_input = torch.randn(3, 3, 256, 256)

    # It's optional to label the input and output layers
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]

    # Use the exporter from torch to convert to onnx
    # model (that has the weights and net arch)
    torch.onnx.export(model, dummy_input, "AICUP.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)

    # vaild_transform = get_vaild_transform()
    #
    # test_image = 'SEG_Train_Datasets/Public_test/output_resize/'
    # test_name = os.listdir(test_image)
    #
    # for idx, n in enumerate(tqdm(test_name)):
    #     path = test_image + n
    #     img = plt.imread(path)
    #     img = vaild_transform(image=img)['image'].unsqueeze(0).to(device)
    #     out2 = F.sigmoid(model2(img)).cpu().detach().numpy()[0].transpose(1, 2, 0)
    #     out = F.sigmoid(model(img)).cpu().detach().numpy()[0].transpose(1, 2, 0)
    #
    #     out[out > 0.5 ] = 255
    #     out[out <= 0.5] = 0
    #     out2[out2 > 0.5] = 255
    #     out2[out2 <= 0.5] = 0
    #     out2 = cv2.resize(out, (1716, 942))
    #     out = cv2.resize(out,(1716, 942))
    #     out3 = cv2.bitwise_and(out,out2)
    #     # print(n.replace('jpg', 'png'))
    #     cv2.imwrite('SEG_Train_Datasets/Public_test/public_output/' + n.replace('jpg', 'png'), out3)



