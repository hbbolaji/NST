import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import copy

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# image size
image_size = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
  transforms.Resize(image_size),
  transforms.ToTensor()
])

def image_loader(image_name):
  image = Image.open(image_name)
  # transform image
  image = loader(image).unsqueeze(dim=0)
  return image.to(device, torch.float)

style_image = image_loader('./dancing.jpg')
content_image = image_loader('./picasso.jpg')

assert style_image.size() == content_image.size()

# image visulization
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
  image = tensor.cpu().clone()
  image = image.squeeze(dim=0)
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.pause(0.005)
  # plt.show()

plt.figure()
imshow(content_image, 'Content Image')

plt.figure()
imshow(style_image, 'Style Image')

# loss functions
class ContentLoss(nn.Module):
  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()
  def forward(self, x):
    self.loss = F.mse_loss(x, self.target)
    return x

def gram_matrix(input):
  a, b, c, d = input.size()
  features  = input.view(a * b, c * d)
  G = torch.mm(features, features.t())
  return G.div(a * b * c * d)

class StyleLoss(nn.Module):
  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()
  def forward(self, x):
    G = gram_matrix(x)
    self.loss = F.mse_loss(G, self.target)
    return x
  
# model
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])