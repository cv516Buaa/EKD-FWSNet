import logging
import torch.nn as nn
import importlib
import copy

logger = logging.getLogger('global')

class ModelBuilder(nn.Module):

    ## build net by name
    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()
        net_name = cfg['name']
        net_sub_name = cfg['subname']
        kwargs = cfg['kwargs']
        net = self.build(net_sub_name, kwargs)
        self.add_module(net_name, net)
    
    def build(self, mname, kwargs):
        module_name, class_name = mname.rsplit('.', 1) 
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        net = cls(**kwargs)
        return net

    def forward(self, input):
        input = copy.copy(input)
        for submodule in self.children():
            output = submodule(input)
        return output