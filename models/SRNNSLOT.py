from typing import Optional
import torch.nn as nn 
import torch.jit as jit
import torch
from torch.nn.functional import layer_norm, softmax
from math import sqrt


class SLOTCell(jit.ScriptModule):
    __constants__ = ['do_embedding', 'single_output', 'multihead']
    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        """
        A single SRNN Slot attention cell. 
        Accepts X of shape [B, L, I] as input
        Uses K slots of size Dh
        Args:
            input_size ([type]): X size
            hidden_size ([type]): Slot size
            num_layers (int, optional): Number of SRNN layers. Defaults to 1.
            hyper_size (int, optional): Number of slots. Defaults to 64.

        """
        super().__init__()
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']
        
        # Slot attention related params
        self.hidden_size = hidden_size
        self.slot_n = hidden_size
        self.k = nn.Linear(input_size, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(input_size, hidden_size)
        self.up_in = nn.Linear(hidden_size, input_size)
        self.s_to_s = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # SRNN related params
        l_list = [nn.Linear(input_size, hyper_size), nn.ReLU()]
        for _ in range(1, num_layers):
            l_list.extend([nn.Linear(hyper_size, hyper_size), nn.ReLU()])
        l_list.append(nn.Linear(hyper_size, hidden_size))
        self.fc = nn.Sequential(*l_list)
        self.fc2 = nn.Linear(input_size, hidden_size)

        
    @jit.script_method
    def init_slots(self, batch_size:int):
        return torch.randn((batch_size, self.slot_n, self.hidden_size))

    @jit.script_method
    def forward(self, x, hidden:Optional[torch.Tensor]=None):
        """
        X->Norm->attn
        slots (hidden) ->Norm->attn->updates->SRNN(updates, prev_slots)
        SRNN(updates, prev_slots): 
        for i in updates:
            outputs.append(updates + rotate(prev_slots))
        return outputs, slots
        """
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape
        #print(f"\nX: {x.shape}")
        
        # Normalize inputs
        with torch.no_grad():
            x_norm = layer_norm(x.detach(), x.shape)
        #print(f"X norm: {x_norm.shape}")

        # Check if slots were inititalized
        if hidden is None:
            hidden = self.init_slots(batch_size).to(x.device)
        # slot operation
        slots = hidden
        updates:torch.Tensor = torch.empty((1))
        k_res:torch.Tensor = torch.empty((1))
        q_res:torch.Tensor = torch.empty((1))
        attn:torch.Tensor = torch.empty((1))
        input_proj:torch.Tensor = torch.empty((1))
        prev_slots:torch.Tensor = torch.empty((1))
        for _ in range(3):
            # Used later in SRNN
            prev_slots = slots
            with torch.no_grad():
                slots = layer_norm(prev_slots, prev_slots.shape)
            k_res = self.k(x_norm)
            q_res = self.q(slots)
            with torch.no_grad():
                attn = softmax((1/sqrt(self.hidden_size))*torch.matmul(k_res, q_res), dim=-1)
            #print(f"K:{k_res.shape}\nQ:{q_res.shape}\nATTN:{attn.shape}\nSLOTS:{slots.shape}")

            input_proj = self.v(x_norm)
            with torch.no_grad():
                updates = torch.mul(attn+1e-4, input_proj)
            updates,_ = self.gru(updates)
            updates = self.up_in(updates)
            slots = self.s_to_s(slots)
            slots = layer_norm(slots, slots.shape)
        #print(f"V:{input_proj.shape}\nUpdates:{updates.shape}")
        # SRNN stage
        b = self.fc(updates)
        if self.multihead:
            sig_alphas = torch.sigmoid(self.fc2(updates))
            b = b * sig_alphas
        
        outputs = [torch.relu(b[:, 0] + torch.roll(prev_slots[:, 0], 1, -1))]
        #print(f"B: {b.shape}\nOut0:{outputs[0].shape}")
        #print(f"Prev:{prev_slots.shape}")
        for i in range(1, seq_len):
            outputs.append(torch.relu(b[:, i] + torch.roll(outputs[-1], 1, -1)))
        
        outputs = torch.stack(outputs, 1)
        outputs = outputs.squeeze(2)
        #print(f"out shape: {outputs.shape}")
        #print(f"hidden shape: {hidden.shape}")
        return outputs, prev_slots


    
class SRNNSLOT(jit.ScriptModule):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = SLOTCell(kwargs['hyper_size'], hidden_size, num_layers, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = SLOTCell(input_size, hidden_size, num_layers, **kwargs)
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
        #print(f"@")
        #print(f"out shape: {outputs.shape}")
        #print(f"hidden shape: {hidden.shape}")
        return outputs, hidden
