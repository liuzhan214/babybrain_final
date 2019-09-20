import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# 卷积归一化激活，常规操作
class ConvBNReLU(nn.Module):
    def __init__(self,in_channels,out_channels,
                    kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),
                    dilation=(1,1,1),bias=False):
        super(ConvBNReLU,self).__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 归一化激活卷积，DenseNet中的设计目的是对所有收集的卷积结果进行归一化
class BNReLUConv(nn.Module):
    def __init__(self,in_channels,out_channels,
                    kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),
                    dilation=(1,1,1),bias=False):
        super(BNReLUConv,self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        
    def forward(self,x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

# 用于轮廓回归，sigmoid输出(0,1)之间的值
class BNSigmoidConv(nn.Module):
    def __init__(self,in_channels,out_channels,
                    kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),
                    dilation=(1,1,1),bias=False):
        super(BNSigmoidConv,self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        
    def forward(self,x):
        x = self.bn(x)
        x = self.sigmoid(x)
        x = self.conv(x)
        return x

# 对来自多层的结果进行BN，RELU，Conv
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

# growth rate控制增长速度，bn_size是bottle-neck size的意思，控制bottle neck的深度
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


# num_layer控制DenseLayer的层数，drop_rate防止过拟合
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

# bottle-neck结构，用于特征精炼
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))

# 包装_Transition模块，用于下采样
class DownSampleBlock(nn.Module):
    def __init__(self,num_input_features, num_output_features):
        super(DownSampleBlock,self).__init__()
        self.downsample = _Transition(num_input_features, num_output_features)
    
    def forward(self,x):
        return self.downsample(x)

# 转置卷积实现上采样
class UpSampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1),output_padding=(0,0,0),bias=False,dilation=1):
        super(UpSampleBlock,self).__init__()
        self.upsample = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride,padding,output_padding,bias=bias,dilation=dilation)
        )

    def forward(self,x):
        o = self.upsample(x)
        return o

# 最终包装的DenseBlock，在_DenseBlock后接一个1x1卷积控制深度
class DenseBlock(nn.Module):
    def __init__(self,num_input_features, num_output_features,num_layers=4,bn_size=4, growth_rate=16, drop_rate=0.5, memory_efficient=False):
        super(DenseBlock, self).__init__()
        self._denseblock = _DenseBlock(num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient)
        self.conv = nn.ModuleList()
        self.conv.append(nn.BatchNorm3d(num_input_features + num_layers*growth_rate))
        self.conv.append(nn.ReLU(inplace=True))
        self.conv.append(nn.Conv3d(num_input_features + num_layers*growth_rate,num_output_features,kernel_size=1,stride=1,padding=0,bias=False))

    def forward(self, init_features):
        o = self._denseblock(init_features)
        for layer in self.conv:
            o = layer(o)
        return o

class DenseUNet(nn.Module):
    def __init__(self):
        super(DenseUNet,self).__init__()
        # fat model，有过拟合风险，此处可以轻松调整模型大小
        # classifier最后的4代表4类分类
        # contour_config最后的1代表轮廓回归输出，结果是(0,1)之间的值
        # first_blcok_config = [64,64,64]
        # unet_config = [56,72,98,114,98,72,56]
        # classifier_config = [64,4]
        # contour_config = [64,1]

        # mid model
        first_blcok_config = [32,32,32]
        unet_config = [40,56,72,88,72,56,40]
        classifier_config = [32,4]
        contour_config = [32,1]
        self.firstblock = nn.Sequential(
            ConvBNReLU(2,first_blcok_config[0]),
            ConvBNReLU(first_blcok_config[0],first_blcok_config[1]),
            ConvBNReLU(first_blcok_config[1],first_blcok_config[2])
        )
        self.downsample_conv = nn.Conv3d(first_blcok_config[-1],first_blcok_config[-1],kernel_size=2,stride=2,padding=0,bias=False)

        self.contraction_path = nn.ModuleList()
        self.expansion_path = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.contour = nn.ModuleList()
        output_channels = []
        output_channels.append(first_blcok_config[-1])
        for i in range(0,len(unet_config)//2):
            output_channel = unet_config[i]
            self.contraction_path.append(DenseBlock(output_channels[-1],output_channel))
            output_channels.append(output_channel)
            self.contraction_path.append(DownSampleBlock(output_channel,output_channel))
            output_channels.append(output_channel)

        for i in range(len(unet_config)//2,len(unet_config)):
            output_channel = unet_config[i]
            self.expansion_path.append(DenseBlock(output_channels[-1],output_channel))
            output_channels.append(output_channel)
            if i < len(unet_config) - 1:
                next_channel = unet_config[i + 1]
            else:
                next_channel = first_blcok_config[-1]
            self.expansion_path.append(UpSampleBlock(output_channel,next_channel))
            output_channels.append(next_channel*2)

        contour_channel = output_channels[-1]
        for i in range(len(classifier_config)):
            output_channel = classifier_config[i]
            if i == len(classifier_config) - 1:
                self.classifier.append(BNReLUConv(output_channels[-1],output_channel,kernel_size=1,stride=1,padding=0,bias=False))
            else:
                self.classifier.append(BNReLUConv(output_channels[-1],output_channel,kernel_size=1,stride=1,padding=0,bias=False))
            output_channels.append(output_channel)

        for i in range(len(contour_config)):
            output_channel = contour_config[i]
            if i == len(classifier_config) - 1:
                self.contour.append(BNSigmoidConv(contour_channel,output_channel,kernel_size=1,stride=1,padding=0,bias=True))
            else:
                self.contour.append(BNReLUConv(contour_channel,output_channel,kernel_size=1,stride=1,padding=0,bias=False))
            contour_channel = output_channel
        
    def forward(self,x):

        x = self.firstblock(x)
        connection_point = [x]
        x = self.downsample_conv(x) 
        for layer in self.contraction_path:
            x = layer(x)
            if isinstance(layer,DenseBlock):
                connection_point.append(x)
        cur = len(connection_point) - 1
        for i,layer in enumerate(self.expansion_path):
            x = layer(x)
            if isinstance(layer,UpSampleBlock):
                x = torch.cat([x,connection_point[cur]],dim=1)
                cur = cur - 1
        x2 = x 
        for layer in self.classifier:
            x = layer(x)

        for layer in self.contour:
            x2 = layer(x2)
        return x,x2

# 测试
# if __name__ == '__main__':
#     x = torch.randn(1,32,32,64,64)
#     db = DenseBlock(num_layers=4,num_input_features=32,num_output_features=64,bn_size=4,growth_rate=16,drop_rate=0)
#     db2 = DenseBlock(num_layers=4,num_input_features=32,num_output_features=64,bn_size=4,growth_rate=16,drop_rate=0)
#     print(db._get_name())
#     y = db(x)
#     print(y.shape)
#     down = DownSampleBlock() 
#     print(down)
#     z = down(y)
#     print(z.shape)
#     up = UpSampleBlock(64,64)
#     print(up)
#     w = up(z)
#     print(w.shape)