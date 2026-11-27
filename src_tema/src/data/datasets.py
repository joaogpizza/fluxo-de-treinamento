"""
Lida com o mapeamento, a leitura e o carregamento dos arquivos
"""

import os
import re
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from configs.basicas import SEED_SPLIT, EXTENSAO_PATCH_MASCARA
from src.data.pareamento import parear_arquivos, ordenar_deterministicamente
from src.data.transformadas import obter_transformacoes


def extrair_id_imagem_mae(caminho_ou_nome: str) -> str:
    """
    Extrai o identificador da imagem-mãe removendo o sufixo `_p\\d+$`
    do nome base do patch.
    """
    nome_base = Path(caminho_ou_nome).stem
    return re.sub(r"_p\d+$", "", nome_base)


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
        extensao_mascara: str = EXTENSAO_PATCH_MASCARA,
        num_particoes: int = 1,
        indice_particao: int = 0,
        seed_particao: int = SEED_SPLIT
    ) -> None:
        if num_particoes < 1:
            raise ValueError("num_particoes deve ser maior ou igual a 1.")
        if not (0 <= indice_particao < num_particoes):
            raise ValueError(
                f"indice_particao ({indice_particao}) deve estar no intervalo [0, {num_particoes - 1}]."
            )

        self.diretorio_imagens = diretorio_imagens
        self.diretorio_mascaras = diretorio_mascaras
        self.extensao_imagem = extensao_imagem
        self.extensao_mascara = extensao_mascara
        self.num_particoes = num_particoes
        self.indice_particao = indice_particao
        self.seed_particao = seed_particao
        self.transformacoes = obter_transformacoes(aplicar_aug)
        self.amostras = self._mapear_arquivos()

        if num_particoes > 1:
            self.amostras = self._filtrar_particao_disjunta(
                amostras=self.amostras,
                num_particoes=num_particoes,
                indice_particao=indice_particao,
                seed_particao=seed_particao,
            )

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

    @staticmethod
    def _filtrar_particao_disjunta(
        amostras: List[Tuple[str, str]],
        num_particoes: int,
        indice_particao: int,
        seed_particao: int,
    ) -> List[Tuple[str, str]]:
        """
        Agrupa os patches por imagem-mãe, ordena deterministicamente as
        imagens-mãe únicas e retorna apenas as amostras da partição solicitada.
        """
        imagens_mae_unicas = sorted({extrair_id_imagem_mae(amostra[0]) for amostra in amostras})
        total_maes = len(imagens_mae_unicas)
        if total_maes < num_particoes:
            raise ValueError(
                f"Número de imagens-mãe ({total_maes}) é menor que num_particoes ({num_particoes})."
            )

        imagens_mae_ordenadas = ordenar_deterministicamente(
            imagens_mae_unicas,
            chave=lambda nome: nome,
            seed=seed_particao,
        )

        tamanho_base, resto = divmod(total_maes, num_particoes)
        inicio = indice_particao * tamanho_base + min(indice_particao, resto)
        fim = inicio + tamanho_base + (1 if indice_particao < resto else 0)
        maes_particao = set(imagens_mae_ordenadas[inicio:fim])

        return [
            amostra for amostra in amostras
            if extrair_id_imagem_mae(amostra[0]) in maes_particao
        ]

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
