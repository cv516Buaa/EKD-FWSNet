"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)

class SE_block(nn.Module):
    def __init__(self, planes):
        super(SE_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)       
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, out):      
        original_out = out
        #print('out.shape',out.shape)
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out  
        #print('out.shape',out.shape)  
        return out


class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM, self).__init__()
        #self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):       
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x      
        return out

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        
        
        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        ########## FA blocks
        self.dropout = nn.Dropout(p=0.1)
        self.drop_blocks = nn.ModuleList([])
        self.drop_blocks_args = []
        for i, data in enumerate(self._blocks_args):
            if i in range(7):
                self.drop_blocks_args.append(data)
        ##########
        group_blocks = [] 
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1
            group_blocks.append(self._blocks)
            ##########
            self._blocks = nn.ModuleList([])
        self.eff_blocks = nn.Sequential(*group_blocks)
        ##########
        ##########
        group_blocks = []
        for block_args in self.drop_blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self.drop_blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: 
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self.drop_blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            group_blocks.append(self.drop_blocks)
            self.drop_blocks = nn.ModuleList([])
        self.drop_eff_blocks = nn.Sequential(*group_blocks)
        ##########
        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        ##########
        self.drop1_conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.drop1_bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.drop2_conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.drop2_bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.drop3_conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.drop3_bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.drop1_fc = nn.Linear(out_channels, self._global_params.num_classes)
        self.drop2_fc = nn.Linear(out_channels, self._global_params.num_classes)
        self.drop3_fc = nn.Linear(out_channels, self._global_params.num_classes)
        ##########

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        idx = 0
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for i in range(self._global_params.branch_point[0]):
            for _, block in enumerate(self.eff_blocks[i]):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / self._global_params.num_MBblock # scale drop connect_rate
                    x = block(x, drop_connect_rate=drop_connect_rate)
                    idx = idx + 1
        aux_out = x
        for i in range(self._global_params.branch_point[0], self._global_params.branch_point[1]):
            for _, block in enumerate(self.eff_blocks[i]):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / self._global_params.num_MBblock # scale drop connect_rate
                    x = block(x, drop_connect_rate=drop_connect_rate)
                    idx = idx + 1
        aux_out_2 = x
        for i in range(self._global_params.branch_point[1], self._global_params.branch_point[2]):
            for _, block in enumerate(self.eff_blocks[i]):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / self._global_params.num_MBblock # scale drop connect_rate
                    x = block(x, drop_connect_rate=drop_connect_rate)
                    idx = idx + 1
        aux_out_3 = x
        for i in range(self._global_params.branch_point[2], len(self.eff_blocks)):
            for _, block in enumerate(self.eff_blocks[i]):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / self._global_params.num_MBblock # scale drop connect_rate
                    x = block(x, drop_connect_rate=drop_connect_rate)
                    idx = idx + 1
        ### aux layer forward
        aux_out = self.dropout(aux_out)
        for i in range(self._global_params.branch_point[0], len(self.eff_blocks)):
            for _, block in enumerate(self.drop_eff_blocks[i]):
                aux_out = block(aux_out)
        aux_out_2 = self.dropout(aux_out_2)
        for i in range(self._global_params.branch_point[1], len(self.eff_blocks)):
            for _, block in enumerate(self.drop_eff_blocks[i]):
                aux_out_2 = block(aux_out_2)
        aux_out_3 = self.dropout(aux_out_3)
        for i in range(self._global_params.branch_point[2], len(self.eff_blocks)):
            for _, block in enumerate(self.drop_eff_blocks[i]):
                aux_out_3 = block(aux_out_3)
        ### output
        output = []
        out = self._swish(self._bn1(self._conv_head(x)))
        out = self._avg_pooling(out)
        if self._global_params.include_top:
            out = out.flatten(start_dim=1)
            out = self._dropout(out)
            out = self._fc(out)
        aux_out = self._swish(self.drop1_bn1(self.drop1_conv_head(aux_out)))
        aux_out = self._avg_pooling(aux_out)
        if self._global_params.include_top:
            aux_out = aux_out.flatten(start_dim=1)
            aux_out = self._dropout(aux_out)
            aux_out = self.drop1_fc(aux_out)
        aux_out_2 = self._swish(self.drop2_bn1(self.drop2_conv_head(aux_out_2)))
        aux_out_2 = self._avg_pooling(aux_out_2)
        if self._global_params.include_top:
            aux_out_2 = aux_out_2.flatten(start_dim=1)
            aux_out_2 = self._dropout(aux_out_2)
            aux_out_2 = self.drop2_fc(aux_out_2)
        aux_out_3 = self._swish(self.drop3_bn1(self.drop3_conv_head(aux_out_3)))
        aux_out_3 = self._avg_pooling(aux_out_3)
        if self._global_params.include_top:
            aux_out_3 = aux_out_3.flatten(start_dim=1)
            aux_out_3 = self._dropout(aux_out_3)
            aux_out_3 = self.drop3_fc(aux_out_3)
        
        output.append(out)
        output.append(aux_out)
        output.append(aux_out_2)
        output.append(aux_out_3)

        out_et = (output[0] + output[1] + output[2] + output[3])/4
        KL_Loss = self.KL_loss(out_et, output[0], T=3)
        KD_Loss = self._global_params.KL_w * KL_Loss
        return output, KD_Loss
    
    def KL_loss(self, teacher, Ss, T=1):
        KL_loss = nn.KLDivLoss()(F.log_softmax(Ss/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        return KL_loss

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, layer_com, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, layer_com, weights_path=weights_path, load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

def efficientnet_b0_nopretrained(pretrained=False, **kwargs):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b0', **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b0', **kwargs)
    return model
def efficientnet_b2_nopretrained(pretrained=False, **kwargs):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b2', **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b2', **kwargs)
    return model
def efficientnet_b4_nopretrained(pretrained=False, **kwargs):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b4', **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b4', **kwargs)
    return model
def efficientnet_b0(pretrained=True, **kwargs):
    layer_combination = [0, 1, 3, 5, 8, 11, 15]
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b0', layer_combination, **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b0', **kwargs)
    return model
def efficientnet_b2(pretrained=True, **kwargs):
    layer_combination = [0, 2, 5, 8, 12, 16, 21]
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b2', layer_combination, **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b2', **kwargs)
    return model
def efficientnet_b4(pretrained=True, **kwargs):
    layer_combination = [0, 2, 6, 10, 16, 22, 30]
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b4', layer_combination, **kwargs)
    else:
        model = EfficientNet.from_name('efficientnet-b4', **kwargs)
    return model