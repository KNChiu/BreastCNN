import argparse
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, AblationCAM, ScoreCAM, LayerCAM	

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

from matplotlib import pyplot as plt

class GradCAMShow():
    def __init__(self, model, checkpoint):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

        target_layer = model.conv3                  # 用在CNN
        # target_layer = model.layer4[-1]      # ResNet
        model.eval()
        self.cam = ScoreCAM(model=model, target_layer=target_layer, use_cuda=True)
        self.cam.batch_size = 8
        
    def img_tensor(self, image_path):
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (640, 640))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])

        return rgb_img, input_tensor

    def get_cam(self, model, image_path):
        rgb_img, input_tensor = self.img_tensor(image_path)

        grayscale_cam = self.cam(input_tensor=input_tensor,
                            target_category=None,
                            aug_smooth=False,
                            eigen_smooth=False)
        
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        return rgb_img, cam_image, cam_mask, cam_gb, gb


if __name__ == '__main__':
    # from model.Custom_model import CNN
    from model.ResNet import ResNet18
    from model.cnn_cbam import CNN
    from grad_cam import GradCAMShow
    import os

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = CNN().to(device)
    image_path = r'data\new_cc\correct'
    save_path = r'CBAM\worst\correct'
    checkpoint = r'CBAM\weight\cnn_cbam7.pth'
    GradCAMShow = GradCAMShow(model, checkpoint)

    # image_path = r'data\new_cc\correct\0a3d5f0d-6d26-4186-a4fe-7d6b2869216e_20210511_105912.png'

    
    i = 1
    for imagepath in os.listdir(image_path):
        print("nume :", i, imagepath) 
        i += 1

        path = os.path.join(image_path, imagepath)

        rgb_img, cam_image, cam_mask, cam_gb, gb = GradCAMShow.get_cam(model, path)

        fig = plt.figure()
        subplot1=fig.add_subplot(2, 2, 1)
        subplot1.imshow(rgb_img)

        subplot2=fig.add_subplot(2, 2, 2)
        subplot2.imshow(cam_image)

        subplot3=fig.add_subplot(2, 2, 3)
        subplot3.imshow(cam_mask)

        subplot4=fig.add_subplot(2, 2, 4)
        subplot4.imshow(cam_gb)

        fig.suptitle("Grad Cam of Best")

        plt.savefig(save_path + "\\" + imagepath)
        plt.close('all')

