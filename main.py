import torch
from torchvision import models, transforms
from torch import optim, nn
from PIL import Image
from torchvision.utils import save_image

weights = models.VGG19_Weights.DEFAULT
auto_transform = weights.transforms()

model = models.vgg19(weights=weights).features
print(model)