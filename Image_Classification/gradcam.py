import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm
import data

class_name = data.class_name
print(class_name)

class ImageSet(Dataset):
    def __init__(self, img, transform = None, class_name = None, label = None):
        self.img = img
        self.label = label
        self.transform = transform
        self.class_name = class_name
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        images = self.img[idx]
        if self.label:
            label = self.label[idx]
            label = class_name[label]
        img = cv2.imread(images)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(img)
        if self.label:
            return image, label
        else:
            return image


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([224, 224]),
    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )

class ConvNext(nn.Module):
    def __init__(self, extraction, head):
        super(ConvNext, self).__init__()
        self.extraction = extraction
        self.head = head
    def forward(self, x):
        x1 = self.extraction(x)
        x2 = self.head(x1)
        return x2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = timm.create_model("convnext_xlarge.fb_in22k", pretrained = True, num_classes = 25)
# model = torch.load("xlarge_distill_best_new.pt", map_location = "cpu")
model = torch.load("best_xlarge_model_v3.pt", map_location = "cpu")
model.to(device)
target_layer = model.stages[-1]
# print(target_layer)
# target_layer = model.extraction[-2][-1]
dataset = pd.read_csv("train.csv")
imageset = ImageSet(img = dataset["upscale_img_path"], transform = transform, class_name = class_name, label = None)
imgloader = DataLoader(imageset, batch_size = 1, shuffle = False)

fig, axes = plt.subplots(10, 10, figsize=(30, 45))
for i, img in enumerate(imgloader):
    # if i < 15: 
    #     continue
    if i >= 100:
        break
    # targets = [ClassifierOutputTarget(label.item())]
    cam = GradCAM(model = model, target_layers = [target_layer])
    cam.batch_size = 1
    grayscale_cam = cam(input_tensor = img)
    grayscale_cam = grayscale_cam[0, :]
    img[img > 1.0] = 1.0
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb = True)
    # display(Image.fromarray(visualization))
    row = (i) // 10 
    col = (i) % 10
    
    # Display the image on the subplot
    axes[row, col].imshow(visualization)
    axes[row, col].axis('off') # Hide axis

# Adjust layout to ensure subplots do not overlap
plt.tight_layout()

# Save the figure containing all subplots
plt.savefig('combined_images_1.png', dpi=300)
# plt.show()