import torch
import torch.nn as nn
from pathlib import Path


class PromptTuner(nn.Module):
    def __init__(self, embed_dim=256, num_prompt_tokens=8, categories=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.categories = categories or []
        
        self.prompt_embeddings = nn.ParameterDict()
        for cat in self.categories:
            self.prompt_embeddings[cat.replace(" ", "_")] = nn.Parameter(
                torch.randn(num_prompt_tokens, embed_dim) * 0.02
            )
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def get_prompt_embedding(self, category):
        key = category.replace(" ", "_")
        if key in self.prompt_embeddings:
            return self.projection(self.prompt_embeddings[key])
        return None
    
    def forward(self, encoded_text, category):
        prompt_emb = self.get_prompt_embedding(category)
        if prompt_emb is None:
            return encoded_text
        
        batch_size = encoded_text.shape[0]
        seq_len = encoded_text.shape[1]
        
        num_tokens_to_add = min(self.num_prompt_tokens, seq_len)
        prompt_emb_trimmed = prompt_emb[:num_tokens_to_add, :]
        prompt_emb_expanded = prompt_emb_trimmed.unsqueeze(0).expand(batch_size, -1, -1)
        
        enhanced_text = encoded_text.clone()
        enhanced_text[:, :num_tokens_to_add, :] = (
            enhanced_text[:, :num_tokens_to_add, :] + prompt_emb_expanded
        )
        
        return enhanced_text


class PromptTunedGroundingDINO(nn.Module):
    def __init__(self, base_model, prompt_tuner):
        super().__init__()
        self.base_model = base_model
        self.prompt_tuner = prompt_tuner
        
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, samples, captions, category=None):
        tokenized = self.base_model.tokenizer(
            captions, padding="longest", return_tensors="pt"
        ).to(samples.device)
        
        from groundingdino.models.GroundingDINO.groundingdino import (
            generate_masks_with_special_tokens_and_transfer_map
        )
        
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.base_model.specical_tokens, self.base_model.tokenizer
        )
        
        if text_self_attention_masks.shape[1] > self.base_model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.base_model.max_text_len, : self.base_model.max_text_len
            ]
            position_ids = position_ids[:, : self.base_model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.base_model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.base_model.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.base_model.max_text_len]
        
        if self.base_model.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized
        
        bert_output = self.base_model.bert(**tokenized_for_encoder)
        encoded_text = self.base_model.feat_map(bert_output["last_hidden_state"])
        
        if category is not None:
            encoded_text = self.prompt_tuner(encoded_text, category)
        
        text_token_mask = tokenized.attention_mask.bool()
        
        if encoded_text.shape[1] > self.base_model.max_text_len:
            encoded_text = encoded_text[:, : self.base_model.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.base_model.max_text_len]
            position_ids = position_ids[:, : self.base_model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.base_model.max_text_len, : self.base_model.max_text_len
            ]
        
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        
        return self.base_model.forward_with_text_dict(samples, text_dict)


def save_prompt_tuner(prompt_tuner, path):
    torch.save({
        "state_dict": prompt_tuner.state_dict(),
        "categories": prompt_tuner.categories,
        "num_prompt_tokens": prompt_tuner.num_prompt_tokens,
        "embed_dim": prompt_tuner.embed_dim
    }, path)


def load_prompt_tuner(path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    prompt_tuner = PromptTuner(
        embed_dim=checkpoint["embed_dim"],
        num_prompt_tokens=checkpoint["num_prompt_tokens"],
        categories=checkpoint["categories"]
    )
    prompt_tuner.load_state_dict(checkpoint["state_dict"])
    return prompt_tuner
