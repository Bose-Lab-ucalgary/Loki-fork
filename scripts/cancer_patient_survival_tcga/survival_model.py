import torch
from torch import nn
from torch.nn import Dropout

class cosine_attention(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.dropout = Dropout(p=dropout)

    def forward(self, patches, weights):
        weights = self.dropout(weights)
        # Use the probabilities as weights to do a weighted sum of the patches
        weighted_patches = weights.unsqueeze(-1) * patches

        output = weighted_patches.sum(dim=1)

        return output[0]


class cos_Surv(nn.Module):
    def __init__(self, fusion=None, embedding_dim=768, n_classes=4, dropout=0):
        super().__init__()
        self.fusion = fusion
        self.n_classes = n_classes

        # Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                *[
                    nn.Linear(embedding_dim * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim),
                    nn.ReLU()
                ]
            )
        else:
            self.mm = None

        # Classifier
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, **kwargs):
        h = kwargs['x_path'][0]  # [768]
        o = kwargs['x_omic']  # [768]

        #print(h.shape, o.shape)
        if self.fusion == 'concat':
            h = self.mm(torch.cat([h, o], axis=0))
        # otherwise, no fusion and only use the image embedding
        
        # Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {'coattn': h}

        return hazards, S, Y_hat, attention_scores
