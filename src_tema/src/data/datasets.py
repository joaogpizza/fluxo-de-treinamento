"""
Lida com o mapeamento, a leitura e o carregamento dos arquivos
"""

import os

from typing import Tuple, List

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from configs.basicas import SEED_SPLIT, EXTENSAO_PATCH_MASCARA
from src.data.pareamento import parear_arquivos, ordenar_deterministicamente
from src.data.transformadas import obter_transformacoes

class HistologiaDataset(Dataset):
    """
    Dataset de segmentação binária sobre os patches gerados por
    scripts/criacao_patches.py.

    Cada amostra é (imagem, máscara): imagem float32 [3, H, W] normalizada e
    máscara float32 [1, H, W] com valores em {0, 1}.
    """
    def __init__(
        self,
        diretorio_imagens: str,
        diretorio_mascaras: str,
        extensao_imagem: str,
        aplicar_aug: bool = False,
        fracao_dados: float = 1.0,
        extensao_mascara: str = EXTENSAO_PATCH_MASCARA
    ) -> None:
        self.diretorio_imagens = diretorio_imagens
        self.diretorio_mascaras = diretorio_mascaras
        self.extensao_imagem = extensao_imagem
        self.extensao_mascara = extensao_mascara
        self.transformacoes = obter_transformacoes(aplicar_aug)
        self.amostras = self._mapear_arquivos()

        if 0.0 < fracao_dados < 1.0:
            num_amostras = max(1, int(len(self.amostras) * fracao_dados))
            # Sorteio determinístico: evita ficar só com os primeiros arquivos
            # em ordem alfabética (que seriam patches das mesmas imagens)
            amostras_embaralhadas = ordenar_deterministicamente(
                self.amostras,
                chave=lambda amostra: os.path.basename(amostra[0]),
                seed=SEED_SPLIT
            )
            self.amostras = amostras_embaralhadas[:num_amostras]

    def _mapear_arquivos(self) -> List[Tuple[str, str]]:
        """
        Mapeia a imagem original com a máscara desta, pelo nome base e em
        ordem determinística.

        Levanta ValueError se alguma imagem não tiver máscara (ou vice-versa)
        e FileNotFoundError se não houver nenhum par.
        """
        pares = parear_arquivos(
            self.diretorio_imagens,
            self.diretorio_mascaras,
            self.extensao_imagem,
            self.extensao_mascara
        )

        return [(par.caminho_imagem, par.caminho_mascara) for par in pares]

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor]:
        caminho_imagem, caminho_mascara = self.amostras[indice]

        with Image.open(caminho_imagem) as imagem_pil:
            imagem_np = np.array(imagem_pil.convert("RGB"))
        with Image.open(caminho_mascara) as mascara_pil:
            mascara_np = np.array(mascara_pil.convert("L"))

        dados_transformados = self.transformacoes(image=imagem_np, mask=mascara_np)

        imagem_tensor = dados_transformados["image"]
        mascara_tensor = dados_transformados["mask"]

        mascara_tensor = (mascara_tensor > 0).float()
        mascara_tensor = mascara_tensor.unsqueeze(0)

        return imagem_tensor, mascara_tensor
