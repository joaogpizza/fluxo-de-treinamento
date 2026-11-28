"""
Implementação da FFG-Unet (Lin et al., "FFG-Unet: A medical image segmentation model
with frequency feature guidance", Pattern Recognition 181 (2027) 114695).

A FFG-Unet é uma U-KAN (src/modelos/ukan.py) com três módulos de guia por frequência:

    RMS    : filtros residuais fixos (sem parâmetros) que geram o mapa de alta frequência (HFM).
    HFEB   : High-Frequency Feature Extraction Block. 3 Swin Transformer Encoders (STE), cada um
             com Swin Transformer Block + Patch Merging, que extraem HF_k a partir do HFM nas
             escalas H/2, H/4 e H/8 (mesmas escalas e canais dos 3 blocos convolucionais do encoder).
    M-sHFGB: Multi-Scale High-Frequency Feature Guidance Block (encoder). Equações (4) e (5):
                 F'_k = CAT(HF_k, F_en_k);  F_en_k_novo = MLP(Conv3x3(F'_k)) + F_en_k
    F-FFB  : Frequency Feature Fusion Block (decoder). Equações (6) a (11): a baixa frequência LF
             vem da aproximação (LL) da DWT de Haar das features do encoder; MF = LF + HF_k, em que
             HF_k é a alta frequência do HFEB correspondente (mesma escala); pesos de canal
             w = Sigmoid(MLP(Conv1x1(Mean(CAT(AvgPool(MF), MaxPool(MF)))))) modulam as features do
             decoder (out = w * F_de).

O bloco KAN tokenizado, a tokenização e os blocos convolucionais são os de src/modelos/ukan.py.
Com usar_m_shfgb=False e usar_f_ffb=False o modelo é equivalente à U-KAN (ablação "Basic" do artigo).

A altura e a largura da entrada devem ser divisíveis por 32 (mesma restrição da U-KAN).

Decisões de implementação onde o artigo é omisso (todas isoladas e fáceis de trocar):
    - RMS: o artigo descreve resíduo -> padronização -> co-ocorrência (Eq. 1 e 2) e usa
      HFM = C_d + C_-d, mas não detalha como isso vira um mapa espacial. Aqui: banco de 3 filtros
      residuais (SRM) -> padronização por imagem -> quantização/truncamento em 2T+1 níveis ->
      cada pixel recebe a frequência (simetrizada d / -d) do seu padrão local de t+1 resíduos
      vizinhos na imagem -> padronização por imagem.
    - Fig. 4 do artigo diz que os pesos do F-FFB são aplicados às features do encoder, enquanto o
      texto e a Eq. 11 dizem decoder. Aqui segue-se o texto/Eq. 11: w modula as features do
      decoder e, depois, a skip connection soma as features do encoder (como na U-KAN).
    - "Mean(.)" da Eq. 10 é a média dos dois descritores (AvgPool e MaxPool) após a concatenação.
    - HFEB: o artigo não diz como o HFM entra no primeiro Swin; aqui há uma projeção linear
      (embedding) do HFM para dim_embedding_hfm canais antes do 1º STE.
"""

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelos.ukan import (
    BlocoConvolucional,
    BlocoConvolucionalDecoder,
    BlocoKAN,
    Tokenizacao,
)


# ---------------------------------------------------------------------------
# Auxiliares de formato
# ---------------------------------------------------------------------------
def _tokens_para_mapa(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, N, C) -> (B, C, H, W)"""
    B = x.shape[0]
    return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


def _mapa_para_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """(B, C, H, W) -> (B, N, C)"""
    _, _, H, W = x.shape
    return x.flatten(2).transpose(1, 2), H, W


def _subir(x: torch.Tensor) -> torch.Tensor:
    """Dobra a resolução (interpolação bilinear) e aplica ReLU (igual à U-KAN)."""
    return F.relu(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


def _haar_ll(x: torch.Tensor) -> torch.Tensor:
    """
    Componente de aproximação (LL) da DWT 2D de Haar ortonormal (Eq. 6):
    LL = (a + b + c + d) / 2 em cada bloco 2x2. Saída com metade da resolução.
    """
    H, W = x.shape[-2:]
    if H % 2 or W % 2:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode="replicate")
    return F.avg_pool2d(x, kernel_size=2) * 2.0


# ---------------------------------------------------------------------------
# RMS: mapa de alta frequência (HFM)
# ---------------------------------------------------------------------------
def _kernels_residuais() -> torch.Tensor:
    """
    Banco de 3 filtros residuais de soma zero, no estilo SRM (Fridrich e Kodovsky, 2012),
    todos preenchidos em 5x5: SQUARE 3x3, SQUARE 5x5 e EDGE 3x3. Saída: (3, 1, 5, 5).
    """
    quadrado_3x3 = torch.tensor([
        [-1.0, 2.0, -1.0],
        [2.0, -4.0, 2.0],
        [-1.0, 2.0, -1.0],
    ]) / 4.0
    quadrado_5x5 = torch.tensor([
        [-1.0, 2.0, -2.0, 2.0, -1.0],
        [2.0, -6.0, 8.0, -6.0, 2.0],
        [-2.0, 8.0, -12.0, 8.0, -2.0],
        [2.0, -6.0, 8.0, -6.0, 2.0],
        [-1.0, 2.0, -2.0, 2.0, -1.0],
    ]) / 12.0
    borda_3x3 = torch.tensor([
        [-1.0, 2.0, -1.0],
        [2.0, -4.0, 2.0],
        [0.0, 0.0, 0.0],
    ]) / 2.0

    kernels = torch.zeros(3, 1, 5, 5)
    kernels[0, 0, 1:4, 1:4] = quadrado_3x3
    kernels[1, 0] = quadrado_5x5
    kernels[2, 0, 1:4, 1:4] = borda_3x3
    return kernels


class FiltrosRMS(nn.Module):
    """
    RMS (Rich Models for Steganalysis) adaptado para gerar o mapa HFM, sem parâmetros treináveis.

    Passos (Eq. 1 e 2 do artigo):
      1. Resíduo R = X_previsto - X (filtros passa-alta de soma zero) sobre a média dos canais.
      2. Resíduo padronizado R_barra (média 0 e desvio 1 por imagem e por filtro).
      3. Quantização em 2T+1 níveis (truncamento em +-T) e padrões de t+1 resíduos vizinhos
         na horizontal.
      4. C_d: frequência de cada padrão d na imagem. HFM(i, j) = C_d + C_-d para o padrão
         d do pixel (i, j) (-d é o padrão com os sinais dos resíduos invertidos).
      5. HFM padronizado por imagem e por filtro.

    Entrada: (B, C, H, W). Saída: (B, num_filtros, H, W), em float32.
    """
    def __init__(self, T: int = 2, t: int = 2, passo_quantizacao: float = 1.0) -> None:
        super().__init__()
        self.T = T
        self.t = t
        self.passo_quantizacao = passo_quantizacao
        self.base = 2 * T + 1
        self.num_padroes = self.base ** (t + 1)
        self.register_buffer("kernels", _kernels_residuais())

    @property
    def num_filtros(self) -> int:
        return self.kernels.shape[0]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().mean(dim=1, keepdim=True)
            residuo = F.conv2d(x, self.kernels, padding=2)

            media = residuo.mean(dim=(2, 3), keepdim=True)
            desvio = residuo.std(dim=(2, 3), keepdim=True)
            residuo = (residuo - media) / (desvio + 1e-6)

            q = torch.round(residuo / self.passo_quantizacao).clamp(-self.T, self.T)
            q = (q + self.T).long()
            return self._cooccorrencia(q)

    def _cooccorrencia(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, n, H, W) inteiros em [0, 2T]. Retorna o mapa HFM (B, n, H, W)."""
        B, n, H, W = q.shape
        t = self.t
        comprimento = W - t

        idx = torch.zeros(B, n, H, comprimento, dtype=torch.long, device=q.device)
        idx_neg = torch.zeros_like(idx)
        for k in range(t + 1):
            fatia = q[..., k:k + comprimento]
            idx = idx + fatia * (self.base ** k)
            idx_neg = idx_neg + (2 * self.T - fatia) * (self.base ** k)

        plano = idx.reshape(B * n, -1)
        plano_neg = idx_neg.reshape(B * n, -1)

        hist = torch.zeros(B * n, self.num_padroes, device=q.device, dtype=torch.float32)
        hist.scatter_add_(1, plano, torch.ones_like(plano, dtype=torch.float32))
        hist = hist / plano.shape[1]

        mapa = hist.gather(1, plano) + hist.gather(1, plano_neg)
        mapa = mapa.reshape(B, n, H, comprimento)
        mapa = F.pad(mapa, (0, t, 0, 0), mode="replicate")

        media = mapa.mean(dim=(2, 3), keepdim=True)
        desvio = mapa.std(dim=(2, 3), keepdim=True)
        return (mapa - media) / (desvio + 1e-6)


# ---------------------------------------------------------------------------
# Swin Transformer (janelas deslocadas), implementado sem timm
# ---------------------------------------------------------------------------
def _particionar_janelas(x: torch.Tensor, janela: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nJanelas, janela * janela, C)"""
    B, H, W, C = x.shape
    x = x.reshape(B, H // janela, janela, W // janela, janela, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, janela * janela, C)


def _reverter_janelas(janelas: torch.Tensor, janela: int, H: int, W: int) -> torch.Tensor:
    """(B * nJanelas, janela * janela, C) -> (B, H, W, C)"""
    C = janelas.shape[-1]
    B = janelas.shape[0] // ((H // janela) * (W // janela))
    x = janelas.reshape(B, H // janela, W // janela, janela, janela, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class AtencaoJanela(nn.Module):
    """Atenção multi-cabeça em janelas, com bias de posição relativa (Swin)."""
    def __init__(self, dim: int, janela: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.escala = (dim // num_heads) ** -0.5

        self.tabela_bias = nn.Parameter(torch.zeros((2 * janela - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.tabela_bias, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(janela), torch.arange(janela), indexing="ij"
        )).flatten(1)
        relativa = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relativa[:, :, 0] += janela - 1
        relativa[:, :, 1] += janela - 1
        relativa[:, :, 0] *= 2 * janela - 1
        self.register_buffer("indice_posicao_relativa", relativa.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor = None) -> torch.Tensor:
        Bj, N, C = x.shape
        qkv = self.qkv(x).reshape(Bj, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        atencao = (q * self.escala) @ k.transpose(-2, -1)
        bias = self.tabela_bias[self.indice_posicao_relativa.reshape(-1)]
        bias = bias.reshape(N, N, -1).permute(2, 0, 1)
        atencao = atencao + bias.unsqueeze(0)

        if mascara is not None:
            nJ = mascara.shape[0]
            atencao = atencao.reshape(Bj // nJ, nJ, self.num_heads, N, N)
            atencao = atencao + mascara.unsqueeze(1).unsqueeze(0)
            atencao = atencao.reshape(-1, self.num_heads, N, N)

        atencao = atencao.softmax(dim=-1)
        x = (atencao @ v).transpose(1, 2).reshape(Bj, N, C)
        return self.proj(x)


class BlocoSwin(nn.Module):
    """Bloco Swin: (S)W-MSA + MLP, com LayerNorm pré-bloco e conexões residuais. Entrada (B, H, W, C)."""
    def __init__(
            self,
            dim: int,
            num_heads: int,
            janela: int = 8,
            deslocamento: int = 0,
            mlp_ratio: float = 2.0
        ) -> None:
        super().__init__()
        self.janela = janela
        self.deslocamento = deslocamento
        self.norma1 = nn.LayerNorm(dim)
        self.atencao = AtencaoJanela(dim, janela, num_heads)
        self.norma2 = nn.LayerNorm(dim)
        oculto = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, oculto), nn.GELU(), nn.Linear(oculto, dim))
        self._cache_mascaras = {}

    def _mascara(self, H: int, W: int, desloc: int, device: torch.device) -> torch.Tensor:
        chave = (H, W, desloc, str(device))
        if chave not in self._cache_mascaras:
            j = self.janela
            img = torch.zeros(1, H, W, 1, device=device)
            fatias = (slice(0, -j), slice(-j, -desloc), slice(-desloc, None))
            cont = 0
            for fh in fatias:
                for fw in fatias:
                    img[:, fh, fw, :] = cont
                    cont += 1
            janelas = _particionar_janelas(img, j).squeeze(-1)
            dif = janelas.unsqueeze(1) - janelas.unsqueeze(2)
            self._cache_mascaras[chave] = dif.masked_fill(dif != 0, -100.0).masked_fill(dif == 0, 0.0)
        return self._cache_mascaras[chave]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        j = self.janela
        if H % j or W % j:
            raise ValueError(f"Altura e largura ({H}x{W}) devem ser divisíveis pela janela do Swin ({j}).")
        desloc = self.deslocamento if min(H, W) > j else 0

        atalho = x
        x = self.norma1(x)
        if desloc > 0:
            x = torch.roll(x, shifts=(-desloc, -desloc), dims=(1, 2))
        janelas = _particionar_janelas(x, j)
        mascara = self._mascara(H, W, desloc, x.device) if desloc > 0 else None
        janelas = self.atencao(janelas, mascara)
        x = _reverter_janelas(janelas, j, H, W)
        if desloc > 0:
            x = torch.roll(x, shifts=(desloc, desloc), dims=(1, 2))

        x = atalho + x
        return x + self.mlp(self.norma2(x))


class FusaoPatches(nn.Module):
    """Patch Merging: junta blocos 2x2 (4C canais), LayerNorm e Linear para dim_saida. (B,H,W,C) -> (B,H/2,W/2,dim_saida)"""
    def __init__(self, dim_entrada: int, dim_saida: int) -> None:
        super().__init__()
        self.norma = nn.LayerNorm(4 * dim_entrada)
        self.reducao = nn.Linear(4 * dim_entrada, dim_saida, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reducao(self.norma(torch.cat([x0, x1, x2, x3], dim=-1)))


class EncoderSwin(nn.Module):
    """
    STE (Swin Transformer Encoder): Swin Transformer Block + Patch Merging.
    O bloco Swin é um par (W-MSA, SW-MSA) por padrão (profundidade = 2), como no Swin original.
    """
    def __init__(
            self,
            dim_entrada: int,
            dim_saida: int,
            janela: int,
            profundidade: int,
            mlp_ratio: float
        ) -> None:
        super().__init__()
        num_heads = dim_entrada // 8 if dim_entrada % 8 == 0 else 1
        self.blocos = nn.ModuleList([
            BlocoSwin(
                dim_entrada, num_heads, janela,
                deslocamento=0 if i % 2 == 0 else janela // 2,
                mlp_ratio=mlp_ratio
            )
            for i in range(profundidade)
        ])
        self.fusao = FusaoPatches(dim_entrada, dim_saida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bloco in self.blocos:
            x = bloco(x)
        return self.fusao(x)


# ---------------------------------------------------------------------------
# Módulos da FFG-Unet
# ---------------------------------------------------------------------------
class HFEB(nn.Module):
    """
    High-Frequency Feature Extraction Block (Eq. 3): HF_k = STE_k(...STE_1(HFM)).

    Entrada: HFM (B, c_hfm, H, W). Saída: lista [HF_1, HF_2, HF_3] com formatos
    (B, C_k, H/2^k, W/2^k), em que C_k = dims_saida[k - 1].
    """
    def __init__(
            self,
            canais_hfm: int,
            dims_saida: Sequence[int],
            dim_embedding: int = 8,
            janela: int = 8,
            profundidade: int = 2,
            mlp_ratio: float = 2.0
        ) -> None:
        super().__init__()
        self.embedding = nn.Linear(canais_hfm, dim_embedding)
        self.norma = nn.LayerNorm(dim_embedding)

        dims_entrada = [dim_embedding] + list(dims_saida[:-1])
        self.estagios = nn.ModuleList([
            EncoderSwin(d_in, d_out, janela, profundidade, mlp_ratio)
            for d_in, d_out in zip(dims_entrada, dims_saida)
        ])
        self.apply(self._inicializar)

    @staticmethod
    def _inicializar(modulo: nn.Module) -> None:
        if isinstance(modulo, nn.Linear):
            nn.init.trunc_normal_(modulo.weight, std=0.02)
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)

    def forward(self, hfm: torch.Tensor) -> List[torch.Tensor]:
        x = self.norma(self.embedding(hfm.permute(0, 2, 3, 1)))
        saidas = []
        for estagio in self.estagios:
            x = estagio(x)
            saidas.append(x.permute(0, 3, 1, 2).contiguous())
        return saidas


class MsHFGB(nn.Module):
    """
    Multi-Scale High-Frequency Feature Guidance Block (Eq. 4 e 5, Fig. 3):
        F' = CAT(HF, F_en);  saída = MLP(Conv3x3(F')) + F_en
    """
    def __init__(self, canais: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * canais, canais, kernel_size=3, padding=1)
        oculto = int(canais * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(canais, oculto), nn.GELU(), nn.Linear(oculto, canais))

    def forward(self, features_encoder: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        mistura = self.conv(torch.cat([hf, features_encoder], dim=1))
        mistura = self.mlp(mistura.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return features_encoder + mistura


class FFFB(nn.Module):
    """
    Frequency Feature Fusion Block (Eq. 6 a 11, Fig. 4).

    forward(features_encoder, hf, features_decoder):
        LF  = Upsample(DWT_Haar(features_encoder).LL)                           (Eq. 6 e 7)
        MF  = LF + HF, com HF = alta frequência do HFEB na mesma escala         (Eq. 8)
        MF' = CAT(AvgPool(MF), MaxPool(MF)) (pooling espacial)                  (Eq. 9)
        w   = Sigmoid(MLP(Conv1x1(Mean(MF'))))                                  (Eq. 10)
        saída = w * features_decoder                                            (Eq. 11)
    """
    def __init__(self, canais: int, reducao: int = 4) -> None:
        super().__init__()
        oculto = max(canais // reducao, 4)
        self.conv1x1 = nn.Conv2d(canais, canais, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(canais, oculto, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(oculto, canais, kernel_size=1),
        )

    def forward(
            self,
            features_encoder: torch.Tensor,
            hf: torch.Tensor,
            features_decoder: torch.Tensor
        ) -> torch.Tensor:
        H, W = features_encoder.shape[-2:]
        lf = F.interpolate(_haar_ll(features_encoder), size=(H, W), mode="bilinear", align_corners=False)
        mf = lf + hf

        media = mf.mean(dim=(2, 3), keepdim=True)
        maximo = mf.amax(dim=(2, 3), keepdim=True)
        descritor = torch.stack([media, maximo], dim=1).mean(dim=1)

        pesos = torch.sigmoid(self.mlp(self.conv1x1(descritor)))
        return pesos * features_decoder


# ---------------------------------------------------------------------------
# FFG-Unet
# ---------------------------------------------------------------------------
class FFGUNet(nn.Module):
    """
    FFG-Unet sobre o backbone da U-KAN (src/modelos/ukan.py).

    Encoder: 3 blocos convolucionais (H/2, H/4, H/8), cada um seguido de um M-sHFGB, e 2 estágios
    com blocos KAN tokenizados (H/16 e H/32).
    Decoder: 2 blocos KAN tokenizados (H/16 e H/8) e blocos convolucionais; nas 3 skip connections
    convolucionais (H/8, H/4 e H/2) o F-FFB modula as features do decoder antes da soma com a skip.

    A altura e a largura da entrada devem ser divisíveis por 32.

    dims_embedding = [128, 160, 256] é o mesmo backbone da U-KAN padrão do repositório. Os canais
    dos 3 estágios convolucionais (e do HFEB) são (d0 // 8, d0 // 4, d0).

    Args:
        janela_swin: tamanho da janela do Swin (8 divide H/4 para qualquer H múltiplo de 32).
        profundidade_swin: nº de blocos Swin por STE (2 = par W-MSA / SW-MSA).
        dim_embedding_hfm: canais do embedding do HFM na entrada do HFEB.
        usar_m_shfgb / usar_f_ffb: ligam/desligam cada módulo (ablação do artigo). Com os dois
            desligados o modelo é equivalente à U-KAN.
    """
    def __init__(
            self,
            canais_entrada: int = 3,
            classes_saida: int = 1,
            dims_embedding: Sequence[int] = (128, 160, 256),
            janela_swin: int = 8,
            profundidade_swin: int = 2,
            dim_embedding_hfm: int = 8,
            usar_m_shfgb: bool = True,
            usar_f_ffb: bool = True
        ) -> None:
        super().__init__()
        self.usar_m_shfgb = usar_m_shfgb
        self.usar_f_ffb = usar_f_ffb

        d0, d1, d2 = dims_embedding
        canais_estagios = (d0 // 8, d0 // 4, d0)

        # Backbone (igual à U-KAN)
        self.encoder1 = BlocoConvolucional(canais_entrada, d0 // 8)
        self.encoder2 = BlocoConvolucional(d0 // 8, d0 // 4)
        self.encoder3 = BlocoConvolucional(d0 // 4, d0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tokenizacao3 = Tokenizacao(d0, d1)
        self.bloco_kan_enc1 = BlocoKAN(d1)
        self.norma_enc1 = nn.LayerNorm(d1)

        self.tokenizacao4 = Tokenizacao(d1, d2)
        self.bloco_kan_enc2 = BlocoKAN(d2)
        self.norma_enc2 = nn.LayerNorm(d2)

        self.decoder1 = BlocoConvolucionalDecoder(d2, d1)
        self.bloco_kan_dec1 = BlocoKAN(d1)
        self.norma_dec1 = nn.LayerNorm(d1)

        self.decoder2 = BlocoConvolucionalDecoder(d1, d0)
        self.bloco_kan_dec2 = BlocoKAN(d0)
        self.norma_dec2 = nn.LayerNorm(d0)

        self.decoder3 = BlocoConvolucionalDecoder(d0, d0 // 4)
        self.decoder4 = BlocoConvolucionalDecoder(d0 // 4, d0 // 8)
        self.decoder5 = BlocoConvolucionalDecoder(d0 // 8, d0 // 8)

        self.saida = nn.Conv2d(d0 // 8, classes_saida, kernel_size=1)

        # Módulos de frequência
        if usar_m_shfgb or usar_f_ffb:
            self.rms = FiltrosRMS()
            self.hfeb = HFEB(
                canais_hfm=self.rms.num_filtros,
                dims_saida=canais_estagios,
                dim_embedding=dim_embedding_hfm,
                janela=janela_swin,
                profundidade=profundidade_swin
            )
        if usar_m_shfgb:
            self.m_shfgb = nn.ModuleList([MsHFGB(c) for c in canais_estagios])
        if usar_f_ffb:
            self.f_ffb = nn.ModuleList([FFFB(c) for c in canais_estagios])

        self._inicializar_pesos_tokens()

    def _inicializar_pesos_tokens(self) -> None:
        """Mesma inicialização da U-KAN para as partes tokenizadas."""
        modulos = [
            self.tokenizacao3, self.tokenizacao4,
            self.bloco_kan_enc1, self.bloco_kan_enc2,
            self.bloco_kan_dec1, self.bloco_kan_dec2,
        ]
        for modulo in modulos:
            for m in modulo.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise ValueError(
                f"A FFG-Unet exige altura e largura divisíveis por 32, mas recebeu {tuple(x.shape[2:])}"
            )

        # Alta frequência: RMS -> HFM -> HFEB (HF_1, HF_2, HF_3 em H/2, H/4 e H/8)
        hf = None
        if self.usar_m_shfgb or self.usar_f_ffb:
            hf = self.hfeb(self.rms(x))

        # Encoder convolucional, com guia de alta frequência (M-sHFGB)
        t1 = self.maxpool(self.encoder1(x))
        if self.usar_m_shfgb:
            t1 = self.m_shfgb[0](t1, hf[0])
        t2 = self.maxpool(self.encoder2(t1))
        if self.usar_m_shfgb:
            t2 = self.m_shfgb[1](t2, hf[1])
        t3 = self.maxpool(self.encoder3(t2))
        if self.usar_m_shfgb:
            t3 = self.m_shfgb[2](t3, hf[2])

        # Encoder com blocos KAN
        out, H, W = self.tokenizacao3(t3)
        out = self.bloco_kan_enc1(out, H, W)
        t4 = _tokens_para_mapa(self.norma_enc1(out), H, W)

        # Gargalo
        out, H, W = self.tokenizacao4(t4)
        out = self.bloco_kan_enc2(out, H, W)
        out = _tokens_para_mapa(self.norma_enc2(out), H, W)

        # Decoder com blocos KAN
        out = _subir(self.decoder1(out)) + t4
        out, H, W = _mapa_para_tokens(out)
        out = self.bloco_kan_dec1(out, H, W)
        out = _tokens_para_mapa(self.norma_dec1(out), H, W)

        out = _subir(self.decoder2(out))
        if self.usar_f_ffb:
            out = self.f_ffb[2](t3, hf[2], out)
        out = out + t3
        out, H, W = _mapa_para_tokens(out)
        out = self.bloco_kan_dec2(out, H, W)
        out = _tokens_para_mapa(self.norma_dec2(out), H, W)

        # Decoder convolucional
        out = _subir(self.decoder3(out))
        if self.usar_f_ffb:
            out = self.f_ffb[1](t2, hf[1], out)
        out = out + t2

        out = _subir(self.decoder4(out))
        if self.usar_f_ffb:
            out = self.f_ffb[0](t1, hf[0], out)
        out = out + t1

        out = _subir(self.decoder5(out))

        return self.saida(out)


if __name__ == "__main__":
    # Teste rápido (a partir de src_tema): python -m src.modelos.ffg_unet
    modelo = FFGUNet()
    total = sum(p.numel() for p in modelo.parameters())
    print(f"FFG-Unet: {total / 1e6:.2f} M de parâmetros")

    entrada = torch.randn(2, 3, 256, 256)
    saida = modelo(entrada)
    print("entrada:", tuple(entrada.shape), "-> saída:", tuple(saida.shape))
    assert saida.shape == (2, 1, 256, 256)

    saida.mean().backward()
    sem_grad = [n for n, p in modelo.named_parameters() if p.requires_grad and p.grad is None]
    print("parâmetros sem gradiente:", sem_grad if sem_grad else "nenhum")
