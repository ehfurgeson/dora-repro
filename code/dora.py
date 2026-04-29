import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALayer(nn.Module):
    def __init__(self, base_layer, rank = 32, alpha = 64):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        # init LoRA matrices (del V)
        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # init magnitude vec (m)
        weight = base_layer.weight.data
        self.m = nn.Parameter(weight.norm(p = 2, dim = 1, keepdim = True))

        # freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        # base weights (V)
        W = self.base_layer.weight

        # LoRA update (del V)
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        W_v = W + lora_update

        # get direction
        norm_W_v = W_v(p = 2, dim = 1, keepdim = True)
        directional_component = W_v / (norm_W_v + 1e-8) # 1e-8 is epsilon added for stability

        # scale by magnitude
        W_dora = self.m * directional_component

        return F.linear(x, W_dora, self.base_layer.bias)
    
    @torch.no_grad()
    def merge_and_unload(self):
        W = self.base_layer.weight
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        W_v = W + lora_update

        norm_W_v = W_v.norm(p=2, dim=1, keepdim=True)
        directional_component = W_v / (norm_W_v + 1e-8)
        W_dora = self.m * directional_component

        self.base_layer.weight.copy_(W_dora)
        return self.base_layer


def apply_dora(model, rank, target_modules = ["q_proj", "v_proj"]):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            dora_layer = DoRALayer(module, rank = rank)
            setattr(model, name, dora_layer)
        else: 
            apply_dora(module, rank, target_modules)
        return model

def merge_and_unload_dora(model):
        for name, module in model.named_children():
            if isinstance(module, DoRALayer):
                standard_linear_layer = module.merge_and_unload()
                setattr(model, name, standard_linear_layer)
            else:
                merge_and_unload_dora(module)
        return model
