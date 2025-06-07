import torch
import torch.nn as nn
import re
import torch.nn.functional as F
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class AttentionMapper(nn.Module):
    def __init__(self, dim_clip, dim_gpt_embedding, prefix_length, add_dropout):
        super(AttentionMapper, self).__init__()
        self.dim_V = dim_clip
        self.num_heads = 8
        self.prefix_length = prefix_length
        self.add_dropout = add_dropout
        self.config = [
            # ( name of param ), [out_size, in_size],
            ('parameter', [prefix_length, dim_clip]),
            ('fc_q_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_k_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_v_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_o_linear', [dim_gpt_embedding, dim_gpt_embedding]),
            ('layer_norm_1', [dim_gpt_embedding]),
            ('layer_norm_2', [dim_gpt_embedding])
        ]
    
        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if 'linear' in name:
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(b)
            elif 'parameter' in name:  # the visual prefix
                param_learn = nn.Parameter(torch.randn(*param), requires_grad=True)
                self.vars.append(param_learn)
            elif 'layer_norm' in name:
                layer_norm_w = nn.Parameter(torch.ones(*param))
                layer_norm_b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(layer_norm_w)
                self.vars.append(layer_norm_b)
    
    def forward(self, clip_x, fast_weights=None, i2l_dict=None):
        
        batch_size, clip_len = clip_x.shape[:2]

        prefix = fast_weights[i2l_dict[0]].unsqueeze(0).expand(batch_size, *fast_weights[i2l_dict[0]].shape)  # I
        x_prefix = torch.cat((prefix, clip_x), dim=1)
                
        Q = F.linear(x_prefix, weight=fast_weights[i2l_dict[1]], bias=fast_weights[i2l_dict[2]])
        K = F.linear(x_prefix, weight=fast_weights[i2l_dict[3]], bias=fast_weights[i2l_dict[4]])
        V = F.linear(x_prefix, weight=fast_weights[i2l_dict[5]], bias=fast_weights[i2l_dict[6]])

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = F.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if self.add_dropout:
            O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[i2l_dict[9]], bias=fast_weights[i2l_dict[10]]) \
            if 'layer_norm_1' in [c[0] for c in self.config] else O

        O = O + F.leaky_relu(F.linear(O, weight=fast_weights[i2l_dict[7]], bias=fast_weights[i2l_dict[8]]))
        if self.add_dropout:
            O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[i2l_dict[11]],  bias=fast_weights[i2l_dict[12]]) \
            if 'layer_norm_2' in [c[0] for c in self.config] else O
        
        O = O[:, :self.prefix_length]

        return O

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == 'attention_mapper':
        return AttentionMapper(config.mm_hidden_size, config.hidden_size, prefix_length=config.prefix_length, add_dropout=config.add_dropout)
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
