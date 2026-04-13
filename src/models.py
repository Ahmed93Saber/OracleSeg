import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import ViTAutoEnc



class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNN, self).__init__()

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # ← LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        ]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # ← Apply LayerNorm here too
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no norm or dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class SimpleNNWithBatchNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNNWithBatchNorm, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_prob)]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, transformer_outputs):
        # transformer_outputs shape: (batch_size, seq_length, hidden_dim)
        scores = torch.matmul(transformer_outputs, self.attention_weights)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * transformer_outputs, dim=1)
        return context_vector

class ViTBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module, unfreeze_last_n: int = 0, out_dim: int = 1):
        super(ViTBinaryClassifier, self).__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks  # nn.ModuleList of TransformerBlock
        self.norm = pretrained_model.norm
        self.attention_pooling = pretrained_model.attention_pooling

        hidden_size = pretrained_model.norm.normalized_shape[0]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, out_dim),
        )

        # Freeze everything by default
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks (if N > 0)
        if unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always train the classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.patch_embedding(x)
        attn_weights = []
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.attention_pooling(x)
        out = self.classifier(x)
        return out


class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model: nn.Module):
        super(ViTEncoder, self).__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks  # nn.ModuleList of TransformerBlock
        self.norm = pretrained_model.norm
        self.attention_pooling = pretrained_model.attention_pooling

        # Freeze everything by default
        for param in self.parameters():
            param.requires_grad = False



    def forward(self, x):
        x = self.patch_embedding(x)
        attn_weights = []
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.attention_pooling(x)
        return x


# Clinical Data Embedding
class ClinicalEmbedding(nn.Module):
    def __init__(self, clinical_input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(clinical_input_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim)
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Attention-based Modality Fusion
class MultimodalAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_modalities = num_modalities

    def forward(self, modality_embeddings):
        # modality_embeddings: list of [batch_size, hidden_dim]
        stacked_embeddings = torch.stack(modality_embeddings, dim=1)  # [batch_size, num_modalities, hidden_dim]

        keys = self.key_proj(stacked_embeddings)  # [batch_size, num_modalities, hidden_dim]
        scores = torch.matmul(keys, self.query)  # [batch_size, num_modalities]

        attention_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [batch_size, num_modalities, 1]

        fused_embedding = torch.sum(attention_weights * stacked_embeddings, dim=1)  # [batch_size, hidden_dim]

        return fused_embedding, attention_weights.squeeze(-1)


# Extended Multimodal Classifier
class MultimodalViTClassifier(nn.Module):
    def __init__(self, pretrained_model, clinical_input_dim, hidden_dim, unfreeze_last_n=0):
        super().__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks
        self.norm = pretrained_model.norm
        self.attention_pooling = pretrained_model.attention_pooling

        self.hidden_dim = hidden_dim

        # Clinical embedding
        self.clinical_embedding = ClinicalEmbedding(clinical_input_dim, hidden_dim)

        # Multimodal attention fusion
        self.modality_fusion = MultimodalAttentionFusion(hidden_dim, num_modalities=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # Freeze parameters by default
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks
        if unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always train the fusion and classifier modules
        for param in self.clinical_embedding.parameters():
            param.requires_grad = True
        for param in self.modality_fusion.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x_in):
        image_3d, clinical_data = x_in

        # ViT branch (3D Image)
        x = self.patch_embedding(image_3d)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        image_embedding = self.attention_pooling(x)  # [batch, hidden_dim]

        # Clinical data branch
        clinical_embedding = self.clinical_embedding(clinical_data)  # [batch, hidden_dim]

        # Attention-based fusion
        fused_embedding, modality_weights = self.modality_fusion([image_embedding, clinical_embedding])

        # Classification
        logits = self.classifier(fused_embedding)

        return logits

# -------- Feature Extractor Wrapper --------
class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_autoenc_model: ViTAutoEnc):
        super(ViTFeatureExtractor, self).__init__()
        self.patch_embedding = vit_autoenc_model.patch_embedding
        self.blocks = vit_autoenc_model.blocks
        self.norm = vit_autoenc_model.norm
        self.attention_pooling = vit_autoenc_model.attention_pooling

    def forward(self, x):
        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        x_attn_pool = self.attention_pooling(x)
        return x_norm, x_attn_pool

