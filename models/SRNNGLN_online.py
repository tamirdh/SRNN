from typing import Optional
import torch.nn as nn
import torch


class Online_SRNN_GLN_Cell(nn.Module):
    __constants__ = ['do_embedding', 'single_output', 'multihead']

    def __init__(self, input_size, hidden_size, num_layers=1, hyper_sizes=(64,), **kwargs):
        super().__init__()
        assert len(
            hyper_sizes) == num_layers, "Layer sizes should match number of layers"
        assert all([i > 1 for i in hyper_sizes]
                   ), f"All hyper layers should be of a size larger than 1\n got:{hyper_sizes}"
       
        layers = list()
        self.fc_in = nn.Linear(input_size, hyper_sizes[0])
        for i in range(num_layers-1):
            # layer i passes l_i(x)*l_(i+1)(x).sigmoid() to l_(i+2)
            layers.extend([nn.Linear(hyper_sizes[i], hyper_sizes[i+1]),
                           nn.Linear(hyper_sizes[i+1], hyper_sizes[i+1])])
        layers.extend([nn.Linear(hyper_sizes[-1], hyper_sizes[-1])]*2)
        self.fc_h = nn.Linear(hyper_sizes[-1], hidden_size)
        self.gln = nn.ModuleList(layers)
        self.fc2 = nn.Linear(input_size, hidden_size)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']

    def forward(self, x, hidden: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape
        assert batch_size == 1, "Online model works with batch size of 1"
        # Linear to GLU layers
        x_out = self.fc_in(x)
        # GLU layer pass through
        b = self.calc_gln(x_out)
        # GLU output to hidden size
        b = self.fc_h(b)

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

    def calc_gln(self, x):
        output = x
        for index, layer in enumerate(self.gln):
            if index % 2 == 1:
                continue
            output = layer(output) * self.gln[index+1](output).sigmoid()
        return output


class Online_SRNN_GLN(nn.Module):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        hyper_sizes = [kwargs["hyper_size"]]*num_layers
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = Online_SRNN_GLN_Cell(
                kwargs['hyper_size'], hidden_size, num_layers, hyper_sizes, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = Online_SRNN_GLN_Cell(
                input_size, hidden_size, num_layers, hyper_sizes, **kwargs)
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
