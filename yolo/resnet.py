import torch
from torchvision.models.resnet import ResNet, Bottleneck



class ResNet_(ResNet):
    #�Զ�����һ��ResNet����ָ������block��layers
    #��һ������������������Ĳ��衣��������˵��һ�����Լ�ѵ��ʱ��
    def __init__(self, block, layers):
        super(ResNet_, self).__init__(block=block, layers=layers)


    #��ǰ����
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        #���в�ز�1,2�������ػ�
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        #������ڲв�ز�3Ҳ�������Ʋ������е�û����
        x = self.layer3(x)
        x = self.maxpool(x)
        return x

    #��ǰ����������ʹ��_forward_impl()������
    def forward(self, x):
        return self._forward_impl(x)


#��һ�δ�����ڴ���һ��_resnet�������������һЩ������������ָ��Ԥѵ��ģ�͵�Ȩ�أ��������������е�̫�鷳��
def _resnet(block, layers, pretrained):
    model = ResNet_(block, layers)
    #����ṩ��pretrained·�����������Ԥѵ����Ȩ�أ�������Ӧ�õ�ģ����
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
