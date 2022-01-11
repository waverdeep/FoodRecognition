import collections
import torch
import torch.nn as nn
import torchvision.models as vision_model


class VGG16Combine(nn.Module):
    def __init__(self, last_node=131072, num_classes=10):
        super(VGG16Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer", vision_model.vgg16_bn(pretrained=True).features),
                ]
            )
        )
        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class ResNET50Combine(nn.Module):
    def __init__(self, last_node=2048, num_classes=10):
        super(ResNET50Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer01", vision_model.resnet50(pretrained=True).conv1),
                    ("feature_extract_layer02", vision_model.resnet50(pretrained=True).bn1),
                    ("feature_extract_layer03", vision_model.resnet50(pretrained=True).relu),
                    ("feature_extract_layer04", vision_model.resnet50(pretrained=True).maxpool),
                    ("feature_extract_layer05", vision_model.resnet50(pretrained=True).layer1),
                    ("feature_extract_layer06", vision_model.resnet50(pretrained=True).layer2),
                    ("feature_extract_layer07", vision_model.resnet50(pretrained=True).layer3),
                    ("feature_extract_layer08", vision_model.resnet50(pretrained=True).layer4),

                ]
            )
        )
        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer09", vision_model.resnet50(pretrained=True).avgpool),
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class ResNET152Combine(nn.Module):
    def __init__(self, last_node=2048, num_classes=10):
        super(ResNET152Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer01", vision_model.resnet152(pretrained=True).conv1),
                    ("feature_extract_layer02", vision_model.resnet152(pretrained=True).bn1),
                    ("feature_extract_layer03", vision_model.resnet152(pretrained=True).relu),
                    ("feature_extract_layer04", vision_model.resnet152(pretrained=True).maxpool),
                    ("feature_extract_layer05", vision_model.resnet152(pretrained=True).layer1),
                    ("feature_extract_layer06", vision_model.resnet152(pretrained=True).layer2),
                    ("feature_extract_layer07", vision_model.resnet152(pretrained=True).layer3),
                    ("feature_extract_layer08", vision_model.resnet152(pretrained=True).layer4),

                ]
            )
        )
        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer09", vision_model.resnet50(pretrained=True).avgpool),
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class WideResNET50_2Combine(nn.Module):
    def __init__(self, last_node=2048, num_classes=10):
        super(WideResNET50_2Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer01", vision_model.wide_resnet50_2(pretrained=True).conv1),
                    ("feature_extract_layer02", vision_model.wide_resnet50_2(pretrained=True).bn1),
                    ("feature_extract_layer03", vision_model.wide_resnet50_2(pretrained=True).relu),
                    ("feature_extract_layer04", vision_model.wide_resnet50_2(pretrained=True).maxpool),
                    ("feature_extract_layer05", vision_model.wide_resnet50_2(pretrained=True).layer1),
                    ("feature_extract_layer06", vision_model.wide_resnet50_2(pretrained=True).layer2),
                    ("feature_extract_layer07", vision_model.wide_resnet50_2(pretrained=True).layer3),
                    ("feature_extract_layer08", vision_model.wide_resnet50_2(pretrained=True).layer4),

                ]
            )
        )
        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer09", vision_model.resnet50(pretrained=True).avgpool),
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class MobileNetV2Combine(nn.Module):
    def __init__(self, last_node=327680, num_classes=10):
        super(MobileNetV2Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer", vision_model.mobilenet_v2(pretrained=True).features),
                ]
            )
        )

        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class DenseNet121Combine(nn.Module):
    def __init__(self, last_node=262144, num_classes=10):
        super(DenseNet121Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer", vision_model.densenet121(pretrained=True).features),
                ]
            )
        )

        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


class SqueezeNet10Combine(nn.Module):
    def __init__(self, last_node=492032, num_classes=10):
        super(SqueezeNet10Combine, self).__init__()
        self.feature_extract_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer", vision_model.squeezenet1_0(pretrained=True).features),
                ]
            )
        )

        self.post_network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("flatten_layer", nn.Flatten()),
                    ("post_layer", nn.Linear(last_node, num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.feature_extract_network(x)
        out = self.post_network(out)
        return out


if __name__ == '__main__':
    sample_data = torch.randn(8, 3, 512, 512)
    test_model = WideResNET50_2Combine()
    print(test_model(sample_data).size())

    # sample_data = torch.randn(8, 3, 512, 512)
    # test_model = ResNET50Combine()
    # sample_out = test_model(sample_data)
    # # print(test_model)
    # print(sample_out.size())
    # test_model = VGG16Combine()
    # sample_data = torch.randn(8, 3, 512, 512)
    # sample_out = test_model(sample_data)
    #
    # print(sample_out)
    #
    # _, predicted = torch.max(sample_out.data, 1)
    # print(predicted)