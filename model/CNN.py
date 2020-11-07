from torchvision import models
from torchsummary import summary
from torch import nn

# vgg = models.vgg16()
# summary(vgg, (3, 224, 224))

resnet50 = models.resnet50()

resnet50 = nn.Sequential(*list(resnet50.children())[:-1])

summary(resnet50, (3, 224, 224))