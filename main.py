import torch
from torchvision import models, transforms
from torch import optim, nn
from PIL import Image
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

weights = models.VGG19_Weights.DEFAULT
auto_transform = weights.transforms()

model = models.vgg19(weights=weights).features

class VGG(nn.Module):
  def __init__(self, weights):
    super(VGG, self).__init__()
    self.chosen_features = ['0', '5', '10', '19', '28']
    self.model = models.vgg19(weights=weights).features[:29]
  
  def forward(self, x):
    features = []
    for (layer_num, layer) in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)
    return features

loader = auto_transform

def load_image(image_name: str):
  image = Image.open(image_name)
  image = loader(image).to(device)
  return image


original_image = load_image('png/content.png')
style_image = load_image('png/style1.png')

generated_image = torch.randn(original_image.shape, device=device, requires_grad=True)

# hyperparament
total_steps = 6800
lr=0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_image], lr=lr)