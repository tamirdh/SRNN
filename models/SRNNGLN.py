from typing import Optional
import torch.nn as nn
import torch


class SRNN_GLN_Cell(nn.Module):
    __constants__ = ['do_embedding', 'single_output', 'multihead']
    def __init__(self, input_size, hidden_size, num_layers=1, hyper_sizes=(64,), **kwargs):
        super().__init__()
        assert len(hyper_sizes) == num_layers, "Layer sizes should match number of layers"
        gated_units_list = list()
        print(f"Input:{input_size}, hyper:{hyper_sizes}")
        for i in range(num_layers):
            if i==0:
                gated_units_list.extend([nn.Linear(input_size, hyper_sizes[0]),
                 nn.Linear(hyper_sizes[0], hyper_sizes[0])])
            else:
                gated_units_list.extend([nn.Linear(hyper_sizes[i-1], hyper_sizes[i]),
                nn.Linear(hyper_sizes[i], hyper_sizes[i])])
        gated_units_list.append(nn.Linear(hyper_sizes[-1], hidden_size))
        self.gln = nn.ModuleList(gated_units_list)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']

    def forward(self, x, hidden: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            _, seq_len = x.shape
        else:
            _, seq_len, _ = x.shape

        b = self.calc_gln(x)

        if self.multihead:
            sig_alphas = torch.sigmoid(self.fc2(x))
            b = b * sig_alphas

        if hidden is not None:
            outputs = [torch.relu(b[:, 0] + torch.roll(hidden, 1, -1))]
        else:
            outputs = [torch.relu(b[:, 0])]

        for i in range(1, seq_len):
            outputs.append(torch.relu(b[:, i] + torch.roll(outputs[-1], 1, -1)))

        outputs = torch.stack(outputs, 1)
        hidden = outputs[:, -1, :]

        outputs = outputs.squeeze(2)

        return outputs, hidden
    
    def calc_gln(self, x):
        out = None
        for index, layer in enumerate(self.gln):
            if out is None:
                out = layer(x) * self.gln[index+1](x).sigmoid()
            elif index % 2 == 0 and index != len(self.gln)-1:
                out = layer(out) * self.gln[index+1](out).sigmoid()
            else:
                continue
        
        return out

class SRNN_GLN(nn.Module):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        hyper_sizes = [int(kwargs['hyper_size']/(i+1)) for i in range(num_layers)]
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = SRNN_GLN_Cell(kwargs['hyper_size'], hidden_size, num_layers, hyper_sizes, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = SRNN_GLN_Cell(input_size, hidden_size, num_layers, hyper_sizes, **kwargs)
            self.do_embedding = False

        self.end_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden: Optional[torch.Tensor] = None):

        if self.do_embedding:
            x = self.embedding(x.squeeze(-1))

        outputs, hidden = self.rnncell(x, hidden)

        outputs = self.end_fc(outputs)
        if self.single_output:
            outputs = outputs[:, -1, :].unsqueeze(1)

        return outputs, hidden
