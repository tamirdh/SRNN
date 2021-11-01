import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.jit as jit


class SRNNMatCell(jit.ScriptModule):
    __constants__ = ['do_embedding', 'single_output', 'multihead']
    
    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        l_list = [nn.Linear(input_size, hyper_size), nn.ReLU()]
        for _ in range(1, num_layers):
            l_list.extend([nn.Linear(hyper_size, hyper_size), nn.ReLU()])
        l_list.append(nn.Linear(hyper_size, hidden_size))
        self.fc = nn.Sequential(*l_list)
        self.fc_end = nn.Linear(hidden_size*hidden_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']
    
    @jit.script_method
    def forward(self, x, hidden: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape
        if hidden is None:
            hidden = torch.zeros((batch_size, self.hidden_size, self.hidden_size), device=x.device)

        outputs = list()
        for i in range(seq_len):
            b = self.fc(x[:,i])

            if self.multihead:
                sig_alphas = torch.sigmoid(self.fc2(x[:,i]))
                b = b * sig_alphas
            if (i%2) ==0:
                hidden = torch.relu(b.unsqueeze(1)+ torch.roll(hidden, 1, -2))
            else:
                hidden = torch.relu(b.unsqueeze(1)+ torch.roll(hidden, 1, -1))
            
            outputs.append(hidden)

        outputs = torch.stack(outputs, 1) # shape= (batch, seq_len, hidden, hidden)
        outputs = self.fc_end(outputs.view((batch_size, seq_len, -1))) # shape= (batch, seq_len, hidden)
        return outputs, hidden
    

class SRNNMat(jit.ScriptModule):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = SRNNMatCell(kwargs['hyper_size'], hidden_size, num_layers, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = SRNNMatCell(input_size, hidden_size, num_layers, **kwargs)
            self.do_embedding = False

        self.end_fc = nn.Linear(hidden_size, output_size)

    @jit.script_method
    def forward(self, x, hidden: Optional[torch.Tensor] = None):

        if self.do_embedding:
            x = self.embedding(x.squeeze(-1))

        outputs, hidden = self.rnncell(x, hidden)

        outputs = self.end_fc(outputs)
        if self.single_output:
            outputs = outputs[:, -1, :].unsqueeze(1)

        return outputs, hidden

