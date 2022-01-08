# coding=utf-8
import math
import torch
from torch.nn.modules import Module
from torch.nn.modules.utils import  _single, _pair, _triple
from torch.nn import init
from torch.nn import Parameter

class ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, sequence_length,kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if(type(padding) == type(1)):
          self.padding=(padding, padding)
        else:
          self.padding = padding
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = torch.nn.Parameter(torch.rand(out_channels, sequence_length, 2, in_channels, kernel_size[0], kernel_size[1]))
        self.weight.requires_grad = True
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.Tensor(torch.zeros(out_channels), dtype= torch.float32)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class ConvFourier(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, sequence_length, stride=1, padding=0, padding_mode ="zeros", dilation=1,bias=True,groups=1):
        self.k_Size = kernel_size
        self.o_channels = out_channels
        self.sequence_length = sequence_length

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.sequence_length = sequence_length
        super(ConvFourier, self).__init__(
        in_channels, out_channels, sequence_length, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias, padding_mode)
    
    def convFourier_forward(self, input , weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.fourierConv(F.pad(input, expanded_padding, mode='circular'),
                                                            weight, self.bias,_pair(0), self.dilation, self.groups)
        return self.fourierConv(input, weight, self.bias, self.out_channels, self.sequence_length, self.dilation, self.groups)
    def forward(self, input):
        return self.convFourier_forward(input, self.weight)

    def fourierConv(self, input, weight, bias, out_channels, sequence_length, dilation, groups):
        batch_size ,in_channels, height, width= input.shape
        input = self.padder(input, self.padding,"constant", 0)
        kernel_size = self.kernel_size
        stride = self.stride[0]
        out_width = ((width - kernel_size[1] + 2*self.padding[1]) // self.stride[1]) + 1
        out_height = ((height - kernel_size[0] + 2*self.padding[0]) // self.stride[0]) + 1
        out_tensor = torch.zeros((batch_size,out_channels,out_height, out_width), dtype=torch.float32).requires_grad_(True);
        for filter_index in range(0, self.out_channels):
            for i in range(0, out_height, stride):
                for j in range(0, out_width, stride):
                    in_tensor_slit = input[ :, :, i:i+kernel_size[0], j:j+kernel_size[0]]
                    out_tensor[:,filter_index, i, j] = self.fourierConvolve(in_tensor_slit, weight, bias,self.sequence_length, filter_index)
        return out_tensor

    def fourierConvolve(self, input, weight, bias, sequence_length, filter_index):
        '''
            Definition :
                    Does the fourier convolution on input of dimension (                                batch_size                            ,channel, height, width)
                                                                       ( filters,2,                   sequence_length                         ,channel, height, width)
                    and weight of dimension                            ( filters,sequence_length,2                                        ,channel, height, width)
        '''
        assert sequence_length <= weight.shape[1]
        out_tensor = torch.tensor(torch.zeros(input.shape, dtype=torch.float32), dtype= torch.float32)
        for sl in range(0,sequence_length):
          alpha = weight[filter_index, sl, 0, :, :, :].view(weight.shape[-3:])
          beta  = (input*sl).cos()
          x =alpha*beta
          out_tensor += x
          out_tensor += weight[filter_index, sl, 1, :, :, :].view(weight.shape[-3:]) *  (input*sl).sin()
        out_tensor = out_tensor.sum(-1)
        out_tensor = out_tensor.sum(-1)
        out_tensor = out_tensor.sum(-1)
        out_tensor += bias[filter_index]/2
        return out_tensor
    
    def padder(self,input, padding, padding_mode="constant", value=0):
        '''
            Definition : Does padding on batch of RGB images i.e. array of
                         4th dimension
            ARGS :
                  input : array of 4 dimension,
                  padding:  Should be a list
                            1. For constant type :- 
                                padding it should be 
                                [0,0,padding_back, padding_front,padding_bottom, 
                                  padding_top, padding_right, padding_left]
                            2. For reflect, replicate mode :-
                                [0,0, padding_right, padding_left]

        '''
        padding = [padding[0], padding[0], padding[1], padding[1]]
        padding = padding[::-1]
        return torch.nn.functional.pad(input, padding, padding_mode, value)






if __name__ == "__main__":

	my_layer = ConvFourier(in_channels = 3,out_channels = 1, kernel_size = 3, sequence_length = 1,padding=[5,3],padding_mode ="zeros", stride = 1)
	a = torch.tensor(torch.rand( 1, 3, 300, 350))

	out  = my_layer(a)