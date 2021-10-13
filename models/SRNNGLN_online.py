from typing import Optional
import torch.nn as nn
import torch


class Online_SRNN_GLN_Cell(nn.Module):
    __constants__ = ['do_embedding', 'single_output', 'multihead']

    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        super().__init__()
        n_classes = kwargs.get("n_classes")
        assert n_classes is not None, "Please supply number of classes to GLN network"
        self.gln = MultiClassGLN(input_size, n_classes, num_layers, hyper_size, halfspaces=2, eps=1e-3)
        self.fc_h = nn.Linear(n_classes, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']

    def forward(self, x, hidden: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, _ = x.shape
            x = x.squeeze(0)
        
        assert batch_size == 1, "Online model works with batch size of 1"
        # GLU layer pass through
        b = self.gln(x, targets)
        # GLU output to hidden size
        b = self.fc_h(b).unsqueeze(0)
    

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

class Online_SRNN_GLN(nn.Module):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = Online_SRNN_GLN_Cell(kwargs["hyper_size"], hidden_size, num_layers, kwargs["hyper_size"], n_classes = output_size)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = self.rnncell = Online_SRNN_GLN_Cell(input_size, hidden_size, num_layers, kwargs["hyper_size"], n_classes = output_size)
            self.do_embedding = False

    def forward(self, x, hidden: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor]=None):

        if self.do_embedding:
            x = self.embedding(x.squeeze(-1))

        outputs, hidden = self.rnncell(x, targets, hidden)

        if self.single_output:
            outputs = outputs[:, -1, :].unsqueeze(1)

        return outputs, hidden


class GLN(nn.Module):
    """
    A Gated Linear Network, composed of multiple layers of GLN neurons.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    def base_predictions(self, z):
        """
        Create base prediction probabilities from the input z by squashing the
        inputs into probabilities.
        """
        epsilon = max(layer.epsilon for layer in self.layers)
        return z.clamp(epsilon, 1 - epsilon)

    def forward(self, x, z):
        """
        Apply the GLN layer-by-layer.
        Args:
            x: an [N x D] Tensor of input probabilities.
            z: an [N x Z] Tensor of side information from the input.
        Returns:
            An [N x K] Tensor of output probabilities.
        """
        for layer in self.layers:
            x = layer(x, z)
        return x

    def forward_grad(self, x, z, targets):
        """
        Apply the GLN on layer-by-layer and compute gradients.
        Args:
            x: an [N x D] Tensor of probabilities from the previous layer.
            z: an [N x Z] Tensor of side information from the input.
            targets: an [N] Tensor of boolean target values.
        Returns:
            An [N x K] Tensor of non-differentiable output probabilities.
        """
        for layer in self.layers:
            x = layer.forward_grad(x, z, targets)
        return x

    def clip_weights(self):
        for layer in self.layers:
            layer.clip_weights()


class Layer(nn.Module):
    """
    A single layer in a Gated Linear Network.
    """

    def __init__(
        self,
        num_side,
        num_inputs,
        num_outputs,
        half_spaces=4,
        epsilon=1e-4,
        weight_clip=10.0,
    ):
        super().__init__()
        assert num_outputs > 0, f"Got {num_outputs} outputs, please supply a number larger than 1"
        self.num_side = num_side
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.half_spaces = half_spaces
        self.epsilon = epsilon
        self.weight_clip = weight_clip
        self.bias_term = 1 - epsilon
        self.lr = 1

        init_gates = torch.randn(num_outputs * half_spaces, num_side)
        init_gates /= (init_gates ** 2).sum(dim=1, keepdim=True).sqrt()
        self.gates = nn.Linear(num_side, num_outputs * half_spaces)
        self.gates.weight.detach().copy_(init_gates)
        self.gates.bias.detach().copy_(torch.randn(half_spaces * num_outputs))

        self.weights = nn.Linear(
            num_inputs + 1, (2 ** half_spaces) * num_outputs, bias=False
        )
        self.weights.weight.detach().fill_(1 / (num_inputs + 1))

    def forward(self, x, z):
        """
        Apply the layer on top of the previous layer's outputs.
        Args:
            x: an [N x D] Tensor of probabilities from the previous layer.
            z: an [N x Z] Tensor of side information from the input.
        Returns:
            An [N x K] Tensor of output probabilities.
        """
        return self._forward(x, z)["probs"]

    def _forward(self, x, z):
        biases = torch.ones_like(x[:, :1]) * self.bias_term
        x = torch.cat([x, biases], dim=-1)
        logit_x = logit(x)
        y = self.weights(logit_x)
        y = y.view(-1, 2 ** self.half_spaces, self.num_outputs)
        gate_choices = self.gate_choices(z)
        y = torch.gather(y, 1, gate_choices).view(-1, self.num_outputs)
        return {
            "logits": y,
            "probs": torch.sigmoid(y).clamp(self.epsilon, 1 - self.epsilon),
            "gate_choices": gate_choices,
        }

    def gate_choices(self, z):
        """
        Compute the gate choices for each neuron.
        Args:
            z: an [N x Z] Tensor of side information from the input.
        Returns:
            An [N x 1 x K] long Tensor of gate choices for each output neuron.
        """
        gate_values = self.gates(
            z).view(-1, self.half_spaces, self.num_outputs)
        gate_bits = gate_values > 0
        gate_choices = torch.zeros_like(gate_bits[:, :1]).long()
        for i, bit in enumerate(gate_bits.unbind(1)):
            gate_choices += bit[:, None].long() * (2 ** i)
        return gate_choices

    def forward_grad(self, x, z, targets):
        """
        Apply the layer and update the gradients of the weights.
        Args:
            x: an [N x D] Tensor of probabilities from the previous layer.
            z: an [N x Z] Tensor of side information from the input.
            targets: an [N] Tensor of boolean target values.
        Returns:
            An [N x K] Tensor of non-differentiable output probabilities.
        """
        forward_out = self._forward(x.detach(), z.detach())
        upstream_grad = self.lr*(forward_out["probs"] - targets.float()[:, None])
        self.lr = min(0.01, self.lr*0.9)
        
        forward_out["logits"].backward(gradient=upstream_grad.squeeze(0).squeeze(0).detach())
        return forward_out["probs"].detach()

    def clip_weights(self):
        for p in self.weights.parameters():
            p.detach().clamp_(-self.weight_clip, self.weight_clip)


def logit(x):
    """
    Inverse of sigmoid.
    """
    return torch.log(x / (1 - x))


class MultiClassGLN(nn.Module):
    """
    A one-versus-all model for discrete classification using binary GLNs.
    """

    def __init__(self, input_size, num_classes, n_layers, hyper_size, halfspaces=4, eps=1e-4, weight_clip=10.0):
        super().__init__()
        self.models = nn.ModuleList([self._create_gln(
            input_size, n_layers, hyper_size, halfspaces, eps, weight_clip) for _ in range(num_classes)])

    def _create_gln(self, input_size, n_layers, hyper_size, halfspaces=3, eps=1e-4, weight_clip=10.0):
        if n_layers == 1:
            layers = [Layer(input_size,hyper_size, 1, halfspaces, eps, weight_clip)]
        else:
            layers = [Layer(input_size, input_size, n_layers *
                            hyper_size, halfspaces, eps, weight_clip)]
            for j in range(1, n_layers-1):
                prev_outputs = layers[j-1].num_outputs
                new_outputs = prev_outputs-hyper_size
                layers.append(Layer(prev_outputs, input_size,
                                    new_outputs, halfspaces, eps, weight_clip))

            layers.append(
                Layer(input_size,layers[-1].num_outputs, 1, halfspaces, eps, weight_clip))
        gln = GLN(*layers)
        return gln

    def forward(self, inputs, targets: Optional[torch.Tensor] = None):
        """
        Apply the models and optionally compute their gradients.
        Args:
            inputs: an [N x D] tensor of inputs.
            targets: an [N] tensor of integer classes.
        Returns:
            An [N x K] tensor of sigmoid probabilities, where K is the number
              of models (output classes). The output is non-differentiable.
        """
        if targets is None:
            outs = [model(model.base_predictions(inputs), inputs)
                    for model in self.models]
        else:
            outs = [
                model.forward_grad(model.base_predictions(inputs), inputs, targets==i)
                for i, model in enumerate(self.models )
            ]
        return torch.cat(outs, dim=-1)

    def clip_weights(self):
        for model in self.models:
            model.clip_weights()
