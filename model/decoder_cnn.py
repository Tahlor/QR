from model.net_builder import make_layers
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel
class DecoderCNN(BaseModel):
    def __init__(self,config):
        super(DecoderCNN, self).__init__(config)
        
        cnn_layer_specs = config['cnn_layer_specs']
        self.cnn_layers, ch_last = make_layers(cnn_layer_specs,dropout=True,norm='batch')

        input_size = config['input_size']
        if type(input_size) is int:
            input_size=[input_size,input_size]

        cnn_output_size=list(input_size)
        scale=1
        for a in cnn_layer_specs:
            if a=='M' or (type(a) is str and a[0]=='D'):
                scale*=2
                cnn_output_size = [i//2 for i in cnn_output_size]
            elif type(a) is str and a[0]=='U':
                scale/=2
                cnn_output_size = [i*2 for i in cnn_output_size]
        cnn_output_size = cnn_output_size[0]*cnn_output_size[1]

        self.num_char_class = config['num_char_class'] if 'num_char_class' in config else 256
        max_message_len = config['max_message_len']

        n_out = 1+self.num_char_class*max_message_len

        fully_connected_layers = config['fully_connected_specs']
        fully_connected_layers = [cnn_output_size*ch_last] + fully_connected_layers + ['FC{}'.format(n_out)]
        self.fc_layers,_ =  make_layers(fully_connected_layers,dropout=True,norm='batch')

        self.cnn_layers=nn.Sequential(*self.cnn_layers)
        self.fc_layers=nn.Sequential(*self.fc_layers)




    def forward(self, x):
        batch_size=x.size(0)
        x=self.cnn_layers(x)
        x=x.view(batch_size,-1)
        out= self.fc_layers(x)
        return out[:,0], out[:,1:].view(batch_size,self.num_char_class,-1)
