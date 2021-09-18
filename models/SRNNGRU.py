from typing import Optional
import torch.nn as nn
import torch 

class SRNNGRUCell(nn.Module):
    """
    Shifting RNN GRU Cell 
    Math:
    SRNN original:
        reference: https://arxiv.org/pdf/2007.07324.pdf
        h_t = sigma(W_p*h_(t-1) + b(x_(t)) = sigma(z_t)
        b(x_t) = f_r(x_t) . sigmoid(W_s*x_t + b_s)
        o_t = s(h_t)
        . is hadamard product
        where:
        sigma is ReLU\\tanh, h_(t-1) in R^(d_h), x in R^(d_i)
        d_h, d_i are the hidden\\input dimensions
        W_p is the permutation matrix (w.l.g off-diagonal matrix)
        b is a network with self gating mechanism
        o_t is the networks output at time t
        s() is an affine layer
    GRU original (fully gated):
        reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit
        z_t = sigmoid(W_z * x_t + U_t*h_(t-1) + b_z)
        r_t = sigmoid(W_r * x_t + U_r*h_(t-1) + b_r)
        y_t = hypertan(W_h * x_t + U_h*[r_t . h_(t-1)] + b_h)
        h_t = (1-z_t) . h_(t-1) + z_t . y_t
        where:
        x_t is input
        h_t is output
        y_t is candidate activation vector
        z_t is update gate vector
        r_t is reset gate vector
        W,U,b are parameter matrices and vector
    
    SRNN GRU Cell (fully gated):
        Combining both the cell types above we achieve
        h_t = sigma(W_p*h_(t-1) + GRUCell(x_(t)) = sigma(z_t)
        o_t = s(h_t)
    """
    __constants__ = ['do_embedding', 'single_output', 'multihead']
    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size, True)
        self.fc_out = nn.Linear(hidden_size, hyper_size)
        self.non_linear = nn.ReLU(False)

    def forward(self, x:torch.Tensor, hidden:Optional[torch.Tensor]= None):
        
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape
        
        # Run x_t through the GRU cell using h_(t-1)
        b = self.gru_cell(x, hidden)
        # Apply permutation on hidden and "+" step
        if hidden is not None:
            outputs = [self.non_linear(b[:, 0]) + torch.roll(hidden, 1, -1)]
        else:
            outputs = [self.non_linear(b[:, 0])]
        for i in range(1, seq_len):
            outputs.append(self.non_linear(b[:,i]) + torch.roll(outputs[-1], 1, -1))
        
        outputs = torch.stack(outputs, 1)
        hidden = outputs[:, -1, :]
        outputs = outputs.squeeze(2)
        
        return outputs, hidden
    
class SRNNGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(1,1)
        self.rnncell = SRNNGRUCell(input_size, hidden_size, num_layers, **kwargs)
        self.do_embed = False

        self.end_fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x:torch.Tensor, hidden:Optional[torch.Tensor]):
        if self.do_embed:
            x = self.embedding(x.squeeze(-1))
        outputs, hidden = self.rnncell(x, hidden)

        outputs = self.end_fc(outputs)

        return outputs, hidden