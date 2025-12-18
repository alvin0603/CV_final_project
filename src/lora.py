import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha
        )
        
        for param in self.original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "query", "value"]
    
    lora_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    lora_layers[name] = LoRALinear(module, rank=rank, alpha=alpha)
                    break
    
    for name, lora_module in lora_layers.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)
    
    return model, lora_layers


def get_lora_parameters(model):
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    return lora_params


def save_lora_weights(model, path):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.clone()
    
    torch.save(lora_state_dict, path)


def load_lora_weights(model, path, device="cuda"):
    lora_state_dict = torch.load(path, map_location=device)
    
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
    
    model.load_state_dict(model_state_dict, strict=False)
    return model


class LoRAGroundingDINO(nn.Module):
    def __init__(self, base_model, rank=4, alpha=1.0):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        
        self.lora_modules = nn.ModuleDict()
        self._inject_lora()
    
    def _inject_lora(self):
        target_names = []
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name.lower() for t in ["query", "key", "value", "q_proj", "k_proj", "v_proj"]):
                    target_names.append(name)
        
        for name in target_names[:20]:
            parts = name.split(".")
            parent = self.base_model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            original = getattr(parent, parts[-1])
            if isinstance(original, nn.Linear):
                lora_linear = LoRALinear(original, rank=self.rank, alpha=self.alpha)
                setattr(parent, parts[-1], lora_linear)
                safe_name = name.replace(".", "_")
                self.lora_modules[safe_name] = lora_linear.lora
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        params = []
        for module in self.lora_modules.values():
            params.extend([module.lora_A, module.lora_B])
        return params
    
    def save_lora(self, path):
        state_dict = {}
        for name, module in self.lora_modules.items():
            state_dict[f"{name}.lora_A"] = module.lora_A.data
            state_dict[f"{name}.lora_B"] = module.lora_B.data
        state_dict["rank"] = self.rank
        state_dict["alpha"] = self.alpha
        torch.save(state_dict, path)
    
    def load_lora(self, path, device="cuda"):
        state_dict = torch.load(path, map_location=device)
        for name, module in self.lora_modules.items():
            if f"{name}.lora_A" in state_dict:
                module.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                module.lora_B.data = state_dict[f"{name}.lora_B"]
