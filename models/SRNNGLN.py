from typing import Optional
import torch.nn as nn
import torch


class SRNNGLN_Cell(nn.Module):
    __constants__ = ['do_embedding', 'single_output', 'multihead']

    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        """
        Single cell of SRNN + GLN sub network

        Args:
            input_size ([type]): size of input to the network 
            hidden_size ([type]): Hidden dimension 
            num_layers (int, optional):  Number of layers in the sub network
            hyper_size (int, optional): Sub network layer size.
        """
        assert num_layers > 0, "Invalid number of layers"
        assert hyper_size > 0, "Invalid layer size"
        super().__init__()
        self.linear = nn.ModuleList()
        self.sigmoid = nn.ModuleList()
        self.linear.append(nn.Linear(input_size, hyper_size))
        self.sigmoid.append(nn.Sequential(nn.Linear(input_size, hyper_size), nn.Sigmoid()))
        for _ in range(1, num_layers):
            self.linear.append(nn.Linear(hyper_size, hyper_size))
            self.sigmoid.append(nn.Sequential(nn.Linear(hyper_size, hyper_size), nn.Sigmoid()))
        self.fc_h = nn.Linear(hyper_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']
    
    def _calc_gln(self, x):
        output = x
        for i in range(len(self.linear)):
            output = self.linear[i](output) * self.sigmoid[i](output)
            output = torch.relu(output)
        return output

    def forward(self, x, hidden: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape

        b = self.fc_h(self._calc_gln(x))
        if self.multihead:
            sig_alphas = torch.sigmoid(self.fc2(x))
            b = b * sig_alphas


        if hidden is not None:
            outputs = [torch.relu(b[:, 0] + torch.roll(hidden, 1, -1))]
        else:
            outputs = [torch.relu(b[:, 0])]

        for i in range(1, seq_len):
            outputs.append(torch.relu(
                b[:, i] + torch.roll(outputs[-1], 1, -1)))

        outputs = torch.stack(outputs, 1)
        hidden = outputs[:, -1, :]

        outputs = outputs.squeeze(2)

        return outputs, hidden


class SRNNGLN(nn.Module):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = SRNNGLN_Cell(
                kwargs['hyper_size'], hidden_size, num_layers, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = SRNNGLN_Cell(
                input_size, hidden_size, num_layers, **kwargs)
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
