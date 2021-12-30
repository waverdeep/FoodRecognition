import collections
import torch
import torch.nn as nn
import torchvision.models as vision_model


class VGG16Combine(nn.Module):
    def __init__(self, last_node=131072, num_classes=10):
        super(VGG16Combine, self).__init__()
        self.network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer", vision_model.vgg16_bn(pretrained=True).features),
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.network(x)
        return out


if __name__ == '__main__':
    test_model = VGG16Combine()
    sample_data = torch.randn(8, 3, 512, 512)
    sample_out = test_model(sample_data)

    print(sample_out.size())
