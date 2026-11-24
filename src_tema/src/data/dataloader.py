"""
Factory para criar os dataloaders de treino, validação e teste
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src.utils.paths import (
    PATH_PATCHES_IMGS_TRAIN, PATH_PATCHES_IMGS_VAL, PATH_PATCHES_IMGS_TEST,
    PATH_PATCHES_GT_TRAIN, PATH_PATCHES_GT_VAL, PATH_PATCHES_GT_TEST
)
from src.data.datasets import HistologiaDataset
from configs.basicas import (
    DEVICE, TAM_BATCH, EXTENSAO_PATCH_IMAGEM, EXTENSAO_PATCH_MASCARA
)

def criar_dataloaders(
    num_workers: int = 0,
    aplicar_aug: bool = False,
    fracao_dados: float = 1.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria os dataloaders de treino, validação e teste do dataset.

    O retorno é na seguinte ordem: dataloader de treino, dataloader de
    validação e dataloader de teste.

    Os patches devem ter sido gerados antes por scripts/criacao_patches.py.
    """

    dataset_treino = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_TRAIN,
        diretorio_mascaras=PATH_PATCHES_GT_TRAIN,
        extensao_imagem=EXTENSAO_PATCH_IMAGEM,
        aplicar_aug=aplicar_aug,
        fracao_dados=fracao_dados,
        extensao_mascara=EXTENSAO_PATCH_MASCARA
    )

    dataset_val = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_VAL,
        diretorio_mascaras=PATH_PATCHES_GT_VAL,
        extensao_imagem=EXTENSAO_PATCH_IMAGEM,
        aplicar_aug=False,
        fracao_dados=fracao_dados,
        extensao_mascara=EXTENSAO_PATCH_MASCARA
    )

    dataset_teste = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_TEST,
        diretorio_mascaras=PATH_PATCHES_GT_TEST,
        extensao_imagem=EXTENSAO_PATCH_IMAGEM,
        aplicar_aug=False,
        fracao_dados=fracao_dados,
        extensao_mascara=EXTENSAO_PATCH_MASCARA
    )

    if len(dataset_treino) < TAM_BATCH:
        raise ValueError(
            f"O treino tem {len(dataset_treino)} patches, menos que o tamanho do batch ({TAM_BATCH}); "
            "com drop_last=True nenhum batch seria gerado. Aumente fracao_dados ou o dataset."
        )

    dl_treino = DataLoader(
        dataset_treino,
        batch_size=TAM_BATCH,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True
    )

    dl_val = DataLoader(
        dataset_val,
        batch_size=TAM_BATCH,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False
    )

    dl_teste = DataLoader(
        dataset_teste,
        batch_size=TAM_BATCH,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False
    )

    return dl_treino, dl_val, dl_teste
