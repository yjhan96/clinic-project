import math
from collections import namedtuple

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
        x = nn.Tanh()(self.alpha * x)
        return self.gamma * x + self.beta


class MultiheadAttention(nn.Module):
    def __init__(
        self, n_heads: int, q_input_dim: int, h_input_dim: int, embed_dim: int
    ):
        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads
        self.q_input_dim = q_input_dim
        self.h_input_dim = h_input_dim
        self.embed_dim = embed_dim
        self.val_dim = embed_dim // n_heads
        self.key_dim = self.val_dim

        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, q_input_dim, self.key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, h_input_dim, self.key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, h_input_dim, self.val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, self.val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # self-attention

        batch_size, setup_len, h_input_dim = h.size()
        _, n_query, q_input_dim = q.size()

        hflat = h.contiguous().view(-1, h_input_dim)
        qflat = q.contiguous().view(-1, q_input_dim)

        shp = (self.n_heads, batch_size, setup_len, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Dimension: (n_heads, batch_size, n_query, setup_len)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            masked_compatibility = compatibility.masked_fill(~mask, -1e9)
            attn = torch.softmax(masked_compatibility, dim=-1)
        else:
            attn = torch.softmax(compatibility, dim=-1)

        # Dimension: (n_heads, batch_sie, n_query, val_dim)
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out


class LogProb(nn.Module):
    def __init__(
        self,
        q_input_dim: int,
        h_input_dim: int,
        embed_dim: int,
        max_output_magnitude: int,
    ):
        super(LogProb, self).__init__()
        self.q_input_dim = q_input_dim
        self.h_input_dim = h_input_dim
        self.embed_dim = embed_dim
        self.max_output_magnitude = max_output_magnitude

        self.norm_factor = 1 / math.sqrt(embed_dim)

        self.W_query = nn.Parameter(torch.Tensor(q_input_dim, self.embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(h_input_dim, self.embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h, mask):
        batch_size, setup_len, h_input_dim = h.size()
        assert q.size(1) == 1
        q_input_dim = q.size(-1)

        # Dimension: (batch_size * setup_len, input_dim)
        hflat = h.contiguous().view(-1, h_input_dim)
        # Dimension:; (batch_size, input_dim)
        qflat = q.contiguous().view(-1, q_input_dim)

        shp = (batch_size, setup_len, -1)
        shp_q = (batch_size, 1, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        unnorm_log_p = self.max_output_magnitude * torch.tanh(compatibility)
        masked_unnorm_log_p = unnorm_log_p.masked_fill(~mask, -1e9)

        # Output dimension: (batch_size, 1, setup_len)
        return torch.log_softmax(masked_unnorm_log_p, dim=-1)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        q_input_dim: int,
        h_input_dim: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.feed_forward_hidden = feed_forward_hidden
        self.mha = MultiheadAttention(
            n_heads,
            q_input_dim=q_input_dim,
            h_input_dim=h_input_dim,
            embed_dim=embed_dim,
        )
        self.first_norm = DyT(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim),
        )
        self.second_norm = DyT(embed_dim)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        h = self.first_norm(q + self.mha(q, h, mask))
        return self.second_norm(h + self.feed_forward(h))


class Encoder(nn.Module):
    def __init__(
        self,
        *,
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
            *(
                MultiHeadAttentionLayer(
                    n_heads, self.embed_dim, self.embed_dim, self.embed_dim
                )
                for _ in range(N)
            )
        )

    def forward(self, nurses, patients, clinics):
        """
        Arguments:
            nurses: Nurse inputs. Dimension: (batch_size, num_nurses, nurse_dim)
            patients: Patient inputs. Dimension: (batch_size, num_patients, patient_dim)
            clinics: Clinic inputs. Dimension: (batch_size, num_clinics, clinic_dim)

        Returns:
            h: embedding of the observation. Dimension: (batch_size, num_nurses + num_patients + num_clinics, embed_dim)
            h_mean: Mean of the embeddings: Dimension: (batch_size, 1, embed_dim)
        """
        nurse_h = self.nurse_embed(nurses)
        patient_h = self.patient_embed(patients)
        clinic_h = self.clinic_embed(clinics)

        h = self.layers(torch.concat([nurse_h, patient_h, clinic_h], dim=1))

        return h, h.mean(dim=1, keepdim=True)


class Decoder(nn.Module):
    def __init__(
        self,
        num_nurses: int,
        num_patients: int,
        num_clinics: int,
        embed_dim: int,
        n_heads: int,
        N: int,
    ):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.nurse_mask = torch.tensor(
            [True for _ in range(num_nurses)]
            + [False for _ in range(num_patients + num_clinics + 1)]
        )
        self.no_op_embedding = nn.Parameter(torch.Tensor(embed_dim))
        self.layers = nn.Sequential(
            *(
                # Input is a concatenation of the mean embedding and a particular nurse embedding.
                MultiHeadAttentionLayer(
                    n_heads, self.embed_dim * 2, self.embed_dim, self.embed_dim * 2
                )
                for _ in range(N)
            )
        )
        self.log_prob = LogProb(
            embed_dim * 2, embed_dim, embed_dim, max_output_magnitude=10
        )
        self.nurse_zero_pad = nn.ZeroPad1d((num_nurses, 0))

    def forward(self, embeddings, embeddings_mean, nurse_idx, valid_actions):
        mask = ~self.nurse_mask & self.nurse_zero_pad(valid_actions)
        batch_size, _setup_len, embed_dim = embeddings.size()
        embeddings = torch.concat(
            [embeddings, self.no_op_embedding.expand(batch_size, 1, embed_dim)], dim=1
        )
        q = torch.concat([embeddings_mean, embeddings[:, [nurse_idx], :]], dim=-1)
        for layer in self.layers:
            q = layer(q, h=embeddings, mask=mask)
        return self.log_prob(q, h=embeddings, mask=mask)


EntityDimension = namedtuple("EntityDimension", ["num_entity", "entity_dim"])


class ClinicNet(nn.Module):
    def __init__(
        self,
        nurse_dim: EntityDimension,
        patient_dim: EntityDimension,
        clinic_dim: EntityDimension,
        embed_dim: int = 128,
        num_heads: int = 8,
        N: int = 6,
    ):
        super(ClinicNet, self).__init__()
        self.nurse_dim = nurse_dim
        self.patient_dim = patient_dim
        self.clinic_dim = clinic_dim

        self.encoder = Encoder(
            nurse_dim=self.nurse_dim.entity_dim,
            patient_dim=self.patient_dim.entity_dim,
            clinic_dim=self.clinic_dim.entity_dim,
            embed_dim=embed_dim,
            n_heads=num_heads,
            N=N,
        )
        self.decoder = Decoder(
            num_nurses=self.nurse_dim.num_entity,
            num_patients=self.patient_dim.num_entity,
            num_clinics=self.clinic_dim.num_entity,
            embed_dim=embed_dim,
            n_heads=num_heads,
            N=N,
        )

    def forward(self, nurses, patients, clinics, nurse_idx, valid_actions):
        embeddings, embeddings_mean = self.encoder(nurses, patients, clinics)
        return self.decoder(embeddings, embeddings_mean, nurse_idx, valid_actions)
