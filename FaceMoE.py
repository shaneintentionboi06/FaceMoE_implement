import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

# ---------------------------------------------------------
# 1. THE MANAGER: Top-k Router
# ---------------------------------------------------------
class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts=3, k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.k = k

    def forward(self, x):
        # Score the experts
        routing_logits = self.router(x) 
        
        # Pick the top 'k' experts
        top_k_logits, expert_indices = torch.topk(routing_logits, self.k, dim=-1)
        
        # Convert to percentages
        routing_weights = F.softmax(top_k_logits, dim=-1)
        
        return routing_weights, expert_indices

# ---------------------------------------------------------
# 2. THE WORKERS: Mixture of Experts Layer
# ---------------------------------------------------------
class MoEMLP(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts=3, k=2):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, k)
        
        # Create the independent expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        routing_weights, expert_indices = self.router(x)
        final_output = torch.zeros_like(x)
        
        # Route tokens to the selected experts
        for i in range(self.router.k):
            current_expert_idx = expert_indices[..., i]
            weight = routing_weights[..., i].unsqueeze(-1)
            
            for exp_id, expert in enumerate(self.experts):
                mask = (current_expert_idx == exp_id)
                if mask.any():
                    expert_output = expert(x[mask])
                    final_output[mask] += expert_output * weight[mask]
                    
        return final_output

# ---------------------------------------------------------
# 3. THE MASTER MODEL: Frozen Swin + FaceMoE Head
# ---------------------------------------------------------
class SwinFaceMoE(nn.Module):
    def __init__(self, num_face_identities, model_name="microsoft/swin-base-patch4-window7-224"):
        super().__init__()
        
        # A. Load the base Swin Transformer
        print("Loading Swin Backbone...")
        self.swin_backbone = SwinModel.from_pretrained(model_name)
        
        # =================================================
        # CRITICAL STEP: FREEZING THE BASE NETWORK
        # This tells PyTorch NOT to train the massive base model,
        # saving 90% of your laptop's memory!
        # =================================================
        for param in self.swin_backbone.parameters():
            param.requires_grad = False
            
        # B. Get the mathematical size of the Swin output (usually 1024 for Swin-Base)
        d_model = self.swin_backbone.config.hidden_size
        
        # C. Attach our custom MoE Layer on top!
        # (This is the ONLY part your laptop will actually train)
        self.moe_layer = MoEMLP(d_model=d_model, hidden_dim=d_model * 4, num_experts=3, k=2)
        
        # D. Final classification layer to guess the Face ID
        self.classifier = nn.Linear(d_model, num_face_identities)

    def forward(self, pixel_values):
        # 1. Pass image through the frozen base model
        outputs = self.swin_backbone(pixel_values=pixel_values)
        
        # Get the feature tokens (Batch, Sequence Length, Hidden Dimension)
        sequence_output = outputs.last_hidden_state
        
        # 2. Pass the tokens into our MoE Layer to fix the blur/degradation
        moe_output = self.moe_layer(sequence_output)
        
        # 3. Pool the tokens together (average them out into one single vector per image)
        pooled_output = moe_output.mean(dim=1)
        
        # 4. Classify the identity
        logits = self.classifier(pooled_output)
        
        return logits