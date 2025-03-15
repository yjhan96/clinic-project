import math

import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class DyT(nn.Module):
    def __init__(self, C: int, init_a: float = 0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_a)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        x = nn.Tanh(self.alpha * x)
        return self.gamma * x + self.beta


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, embed_dim: int):
        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = embed_dim // n_heads
        self.key_dim = self.val_dim

        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, self.key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, self.key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, self.val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, self.val_dim, embed_dim))

        # TODO: Is this right?
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None):
        if h is None:
            h = q  # self-attention

        batch_size, setup_len, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, setup_len, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out


class LogProb(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, max_output_magnitude: int):
        super(LogProb, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_output_magnitude = max_output_magnitude

        self.norm_factor = 1 / math.sqrt(embed_dim)

        self.W_query = nn.Parameter(torch.Tensor(input_dim, self.embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(input_dim, self.embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h):
        batch_size, setup_len, input_dim = h.size()
        assert q.size(1) == 1

        # Dimension: (batch_size * setup_len, input_dim)
        hflat = h.contiguous().view(-1, input_dim)
        # Dimension:; (batch_size, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (batch_size, setup_len, -1)
        shp_q = (batch_size, 1, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        unnorm_log_p = self.max_output_magnitude * torch.tanh(compatibility)

        # Output dimension: (batch_size, 1, setup_len)
        return torch.log_softmax(unnorm_log_p, dim=-1)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self, n_heads: int, embed_dim: int, feed_forward_hidden: int = 512):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiheadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
            ),
            DyT(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
            ),
            DyT(embed_dim),
        )


class Encoder(nn.Module):
    def __init__(
        self,
        nurse_dim: int,
        patient_dim: int,
        clinic_dim: int,
        embed_dim: int,
        n_heads: int,
        N: int,
    ):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.nurse_embed = nn.Linear(nurse_dim, self.embed_dim)
        self.patient_embed = nn.Linear(patient_dim, self.embed_dim)
        self.clinic_embed = nn.Linear(clinic_dim, self.embed_dim)

        self.layers = nn.Sequential(
            *(MultiHeadAttentionLayer(n_heads, self.embed_dim) for _ in range(N))
        )

    def forward(self, nurses, patients, clinics):
        nurse_h = self.nurse_embed(nurses)
        patient_h = self.patient_embed(patients)
        clinic_h = self.clinic_embed(clinics)

        h = self.layers(torch.concat([nurse_h, patient_h, clinic_h], dim=1))

        return h, h.mean(dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        n_heads: int,
        N: int,
    ):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.layers = [MultiheadAttention(n_heads, self.embed_dim) for _ in range(N)]
        self.log_prob = LogProb(embed_dim, embed_dim, max_output_magnitude=10)

    def forward(self, embeddings, embeddings_mean, nurse_idx):
        q = torch.conat([embeddings_mean, embeddings[:, [nurse_idx], :]], dim=1)
        for layer in self.layers:
            q = layer(q, h=embeddings)
        return self.log_prob(q, h=embeddings)


class ClinicNet(nn.Module):
    def __init__(
        self,
        num_nurses: int,
        num_patients: int,
        num_clinics: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        N: int = 6,
    ):
        super(ClinicNet, self).__init__()
        self.num_nurses = num_nurses
        self.num_patients = num_patients
        self.num_clinics = num_clinics

        self.encoder = Encoder(
            num_nurses, num_patients, num_clinics, embed_dim, num_heads, N
        )
        self.decoder = Decoder(embed_dim, num_patients + num_clinics + 1, num_heads, N)

    def forward(self, input, nurse_idx):
        embeddings, embeddings_mean = self.encoder(input)
        return self.decoder(embeddings, embeddings_mean, nurse_idx)
