import torch
from torchvision import models, transforms
from torch import optim, nn
from PIL import Image
from torchvision.utils import save_image

weights = models.VGG19_Weights.DEFAULT
auto_transform = weights.transforms()

model = models.vgg19(weights=weights).features
print(model)

class VGG(nn.Module):
  def __init__(self, weights):
    super(VGG, self).__init__()
    self.chosen_features = ['0', '5', '10', '19', '28']
    self.model = models.vgg19(weights=weights).features[:29]
  
  def forward(self, x):
    features = []