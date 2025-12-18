import torch
import torch.nn as nn
import math


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class LoRAWithAdapter(nn.Module):
    def __init__(self, original_linear, lora_rank=4, lora_alpha=1.0, adapter_bottleneck=64):
        super().__init__()
        self.original_linear = original_linear
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_scaling = lora_alpha / lora_rank
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.adapter = Adapter(out_features, adapter_bottleneck)
        
        for param in self.original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        base_output = self.original_linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.lora_scaling
        combined = base_output + lora_output
        output = self.adapter(combined)
        return output


class LoRAAdapterGroundingDINO(nn.Module):
    def __init__(self, base_model, lora_rank=4, lora_alpha=1.0, adapter_bottleneck=64):
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.adapter_bottleneck = adapter_bottleneck
        
        self.hybrid_modules = nn.ModuleDict()
        self._inject_hybrid()
    
    def _inject_hybrid(self):
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
                hybrid_module = LoRAWithAdapter(
                    original, 
                    lora_rank=self.lora_rank, 
                    lora_alpha=self.lora_alpha,
                    adapter_bottleneck=self.adapter_bottleneck
                )
                setattr(parent, parts[-1], hybrid_module)
                safe_name = name.replace(".", "_")
                self.hybrid_modules[safe_name] = hybrid_module
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        params = []
        for module in self.hybrid_modules.values():
            params.append(module.lora_A)
            params.append(module.lora_B)
            params.extend(module.adapter.parameters())
        return params
    
    def save_hybrid(self, path):
        state_dict = {}
        for name, module in self.hybrid_modules.items():
            state_dict[f"{name}.lora_A"] = module.lora_A.data
            state_dict[f"{name}.lora_B"] = module.lora_B.data
            for k, v in module.adapter.state_dict().items():
                state_dict[f"{name}.adapter.{k}"] = v
        state_dict["lora_rank"] = self.lora_rank
        state_dict["lora_alpha"] = self.lora_alpha
        state_dict["adapter_bottleneck"] = self.adapter_bottleneck
        torch.save(state_dict, path)
    
    def load_hybrid(self, path, device="cuda"):
        state_dict = torch.load(path, map_location=device)
        for name, module in self.hybrid_modules.items():
            if f"{name}.lora_A" in state_dict:
                module.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                module.lora_B.data = state_dict[f"{name}.lora_B"]
            
            adapter_state = {}
            for k, v in state_dict.items():
                if k.startswith(f"{name}.adapter."):
                    adapter_key = k.replace(f"{name}.adapter.", "")
                    adapter_state[adapter_key] = v
            if adapter_state:
                module.adapter.load_state_dict(adapter_state)
