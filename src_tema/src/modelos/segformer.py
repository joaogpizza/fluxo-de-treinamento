import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class OverlappedPatchMerging(nn.Module):
    """
    Módulo para realizar a mesclagem de patches com sobreposição.
    Substitui o patch merging convencional para manter a continuidade local ao reduzir a dimensionalidade.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Transforma a imagem ou mapa de características em patches com sobreposição,
        retornando a sequência achatada juntamente com suas novas dimensões espaciais.
        """
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class EfficientSelfAttention(nn.Module):
    """
    Módulo de autoatenção eficiente que utiliza um rácio de redução espacial
    para diminuir o comprimento da sequência das Chaves (K) e Valores (V), 
    reduzindo a complexidade computacional para O(N^2/R).
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Calcula a atenção de forma eficiente aplicando a redução espacial aos tensores K e V.
        """
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_2d = x.transpose(1, 2).reshape(b, c, h, w)
            x_reduced = self.sr(x_2d).flatten(2).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
        else:
            x_reduced = x

        kv = self.kv(x_reduced).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k = kv[0]
        v = kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x_out = self.proj(x_out)
        return x_out

class MixFFN(nn.Module):
    """
    Rede Feed-Forward Mista (Mix-FFN) que vaza informações de posição diretamente 
    na rede usando uma convolução 3x3 depth-wise, eliminando a codificação posicional de tamanho fixo.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Aplica projeção linear, convolução de profundidade, ativação GELU e projeção final.
        """
        x = self.fc1(x)
        b, n, c = x.shape
        x_2d = x.transpose(1, 2).reshape(b, c, h, w)
        x_2d = self.dwconv(x_2d)
        x_2d = self.act(x_2d)
        x_out = x_2d.flatten(2).transpose(1, 2)
        x_out = self.fc2(x_out)
        return x_out

class TransformerBlock(nn.Module):
    """
    Bloco Transformer base contendo Efficient Self-Attention, Mix-FFN e conexões residuais padronizadas.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, sr_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=dim * mlp_ratio, out_features=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Processa a entrada sequencialmente através das camadas de atenção e feed-forward com adições residuais.
        """
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x

class MiTEncoder(nn.Module):
    """
    Codificador Mix Transformer (MiT) hierárquico que extrai características espaciais
    nas resoluções 1/4, 1/8, 1/16 e 1/32 da imagem original.
    """
    def __init__(self, in_channels: int, embed_dims: List[int], num_heads: List[int], mlp_ratios: List[int], depths: List[int], sr_ratios: List[int]) -> None:
        super().__init__()
        self.patch_embed1 = OverlappedPatchMerging(in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3)
        self.block1 = nn.ModuleList([TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.patch_embed2 = OverlappedPatchMerging(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.block2 = nn.ModuleList([TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.patch_embed3 = OverlappedPatchMerging(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.block3 = nn.ModuleList([TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.patch_embed4 = OverlappedPatchMerging(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1)
        self.block4 = nn.ModuleList([TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Executa os quatro estágios de mesclagem e atenção, armazenando e retornando as características hierárquicas.
        """
        outs = []

        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x_out1 = self.norm1(x).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        outs.append(x_out1)

        x, h, w = self.patch_embed2(x_out1)
        for blk in self.block2:
            x = blk(x, h, w)
        x_out2 = self.norm2(x).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        outs.append(x_out2)

        x, h, w = self.patch_embed3(x_out2)
        for blk in self.block3:
            x = blk(x, h, w)
        x_out3 = self.norm3(x).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        outs.append(x_out3)

        x, h, w = self.patch_embed4(x_out3)
        for blk in self.block4:
            x = blk(x, h, w)
        x_out4 = self.norm4(x).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        outs.append(x_out4)

        return outs

class AllMLPDecoder(nn.Module):
    """
    Decodificador leve All-MLP corrigido para estabilidade de gradientes (Dice Loss).
    Utiliza convoluções 1x1, BatchNorm2d e ReLU no lugar de projeções lineares puras,
    simulando o comportamento de MLP de forma estabilizada sobre mapas de características 2D.
    """
    def __init__(self, embed_dims: List[int], decoder_dim: int, num_classes: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)
        self.c2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
        self.c3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
        self.c4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        self.pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        _, _, h, w = c1.shape

        _c1 = self.c1(c1)
        
        _c2 = F.interpolate(self.c2(c2), size=(h, w), mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.c3(c3), size=(h, w), mode='bilinear', align_corners=False)
        _c4 = F.interpolate(self.c4(c4), size=(h, w), mode='bilinear', align_corners=False)

        f_concat = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        f_fused = self.fuse(f_concat)
        
        mask = self.pred(f_fused)

        return mask

class SegFormer(nn.Module):
    """
    Arquitetura principal do SegFormer para segmentação semântica, integrando
    o codificador MiT hierárquico e o decodificador All-MLP, suportando inferência
    em tensores de resolução arbitrária incluindo 256x256.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        decoder_dim: int = 256
    ) -> None:
        super().__init__()
        self.encoder = MiTEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios
        )
        self.decoder = AllMLPDecoder(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai características multiescala, decodifica a máscara em resolução 1/4
        e finaliza fazendo o upsampling da máscara para as dimensões exatas da entrada original.
        """
        features = self.encoder(x)
        mask = self.decoder(features)
        mask_out = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        return mask_out
