import torch
from torchvision.models.resnet import ResNet, Bottleneck



class ResNet_(ResNet):
    #自定义了一个ResNet可以指定它的block和layers
    #这一步不是论文中所必需的步骤。但是网上说这一步可以简化训练时长
    def __init__(self, block, layers):
        super(ResNet_, self).__init__(block=block, layers=layers)


    #向前传播
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        #将残差池层1,2进行最大池化
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        #好像对于残差池层3也进行类似操作，有点没看懂
        x = self.layer3(x)
        x = self.maxpool(x)
        return x

    #向前传播。这里使用_forward_impl()方法。
    def forward(self, x):
        return self._forward_impl(x)


#这一段大概是在创建一个_resnet对象，配置里面的一些参数。并且能指定预训练模型的权重，但看起来好像有点太麻烦了
def _resnet(block, layers, pretrained):
    model = ResNet_(block, layers)
    #如果提供了pretrained路径，它会加载预训练的权重，并将其应用到模型中
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model


def resnet_1024ch(pretrained=None) -> ResNet:
    resnet = _resnet(Bottleneck, [3, 4, 6, 3], pretrained)
    return resnet


if __name__ == '__main__':
    x = torch.randn([1, 3, 448, 448])
    net = resnet_1024ch('resnet50-19c8e357.pth')
    print(net)

    y = net(x)
    print(y.size())
