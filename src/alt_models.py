import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projection layers for queries, keys, and values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, query_modality, key_value_modality):
        """
        Args:
            query_modality: [batch_size, seq_len_q, hidden_dim] or [batch_size, hidden_dim]
            key_value_modality: [batch_size, seq_len_kv, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            attended_features: [batch_size, seq_len_q, hidden_dim] or [batch_size, hidden_dim]
        """
        batch_size = query_modality.shape[0]

        # Handle single vector inputs by adding sequence dimension
        if len(query_modality.shape) == 2:
            query_modality = query_modality.unsqueeze(1)  # [batch, 1, hidden_dim]
            squeeze_output = True
        else:
            squeeze_output = False

        if len(key_value_modality.shape) == 2:
            key_value_modality = key_value_modality.unsqueeze(1)

        seq_len_q = query_modality.shape[1]
        seq_len_kv = key_value_modality.shape[1]

        # Project to Q, K, V
        Q = self.q_proj(query_modality)  # [batch, seq_len_q, hidden_dim]
        K = self.k_proj(key_value_modality)  # [batch, seq_len_kv, hidden_dim]
        V = self.v_proj(key_value_modality)  # [batch, seq_len_kv, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_len_q, head_dim]

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )

        # Output projection
        attended = self.out_proj(attended)

        # Residual connection and layer norm
        attended = self.layer_norm1(attended + query_modality)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(attended)
        attended = self.layer_norm2(attended + ffn_output)

        # Remove sequence dimension if input was single vector
        if squeeze_output:
            attended = attended.squeeze(1)

        return attended, 1


class ClinicalEmbedding(nn.Module):
    def __init__(self, clinical_input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(clinical_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)


class SequenceCrossAttentionClassifier(nn.Module):
    def __init__(self, pretrained_model, clinical_input_dim, hidden_dim,
                 num_heads=8, unfreeze_last_n=0):
        super().__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks
        self.norm = pretrained_model.norm
        self.hidden_dim = hidden_dim

        # Clinical embedding
        self.clinical_embedding = ClinicalEmbedding(clinical_input_dim, hidden_dim)

        # Cross-attention (clinical attends to image patches)
        self.cross_attention = CrossAttentionFusion(hidden_dim, num_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*hidden_dim, 1)
        )

        # Freeze parameters by default
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks
        if unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always train new components
        for param in self.clinical_embedding.parameters():
            param.requires_grad = True
        for param in self.cross_attention.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x_in):
        image_3d, clinical_data = x_in

        # ViT branch - keep full sequence
        x = self.patch_embedding(image_3d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [batch, seq_len, hidden_dim]

        # Clinical embedding
        clinical_embedding = self.clinical_embedding(clinical_data)  # [batch, hidden_dim]

        # Cross-attention: clinical queries attend to image patches
        attended_clinical, attention_weights = self.cross_attention(clinical_embedding, x)  # [batch, hidden_dim]

        # Classification
        logits = self.classifier(attended_clinical)

        return logits


class GuidedCrossAttention(nn.Module):
    """
    Guided attention where the clinical modality guides (queries)
    and the image modality provides context (keys/values).
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attention = CrossAttentionFusion(hidden_dim, num_heads, dropout)

    def forward(self, clinical_embedding, image_embeddings):
        # Clinical modality = query
        # Image modality = keys/values
        guided_features, attn_weights = self.cross_attention(
            query_modality=clinical_embedding,
            key_value_modality=image_embeddings
        )
        return guided_features, attn_weights

class SequenceGuidedAttentionClassifier(nn.Module):
    def __init__(self, pretrained_model, clinical_input_dim, hidden_dim,
                 num_heads=8, unfreeze_last_n=0):
        super().__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks
        self.norm = pretrained_model.norm

        # Clinical embedding branch
        self.clinical_embedding = ClinicalEmbedding(clinical_input_dim, hidden_dim)

        # Guided attention: clinical guides image
        self.guided_attention = GuidedCrossAttention(hidden_dim, num_heads)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # Freeze pretrained layers (same logic as before)
        for param in self.parameters():
            param.requires_grad = False
        if unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.clinical_embedding.parameters():
            param.requires_grad = True
        for param in self.guided_attention.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x_in):
        image_3d, clinical_data = x_in

        # ViT image sequence
        x = self.patch_embedding(image_3d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [batch, seq_len, hidden_dim]

        # Clinical embedding
        clinical_emb = self.clinical_embedding(clinical_data)  # [batch, hidden_dim]

        # Guided attention
        guided_features, attn_weights = self.guided_attention(clinical_emb, x)

        # Classification
        logits = self.classifier(guided_features)
        return logits, attn_weights




