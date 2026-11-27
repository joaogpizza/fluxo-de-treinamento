"""Testes unitários para particionamento disjunto e estratégias de ensemble."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from configs.basicas import DEVICE, ESTRATEGIAS_AGREGACAO_ENSEMBLE, TAM_PATCH
from src.avaliacao.avaliador import avaliar_ensemble_checkpoints
from src.data import dataloader as dataloader_mod
from src.data.datasets import HistologiaDataset, extrair_id_imagem_mae
from src.modelos.ensemble import EnsembleEndToEnd, EnsembleInferencia
from src.modelos.modelo_factory import pegar_modelo
from src.otimizadores.muon import MuonAdamW
from src.utils import tabelas as tabelas_mod
from src.utils.checkpoints import salvar_checkpoint


class _ModeloConstante(nn.Module):
    """Modelo auxiliar determinístico que retorna logits constantes `[B, 1, H, W]`."""

    def __init__(self, valor_logit: float = 0.0) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor([[valor_logit]], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return self.param.view(1, 1, 1, 1).expand(b, 1, h, w).clone()


class _ModeloLinear2D(nn.Module):
    """Modelo mínimo com parâmetro 2D e 1D para validar fluxo de gradientes e MuonAdamW."""

    def __init__(self, escala: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 1, bias=True)
        with torch.no_grad():
            self.proj.weight.fill_(escala)
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] -> [B, H, W, 3] -> [B, H, W, 1] -> [B, 1, H, W]
        saida = self.proj(x.permute(0, 2, 3, 1))
        return saida.permute(0, 3, 1, 2)


def _criar_patches_sinteticos(
    dir_imgs: Path,
    dir_masks: Path,
    nomes_patches: list[str],
    tamanho: int = 16,
) -> None:
    dir_imgs.mkdir(parents=True, exist_ok=True)
    dir_masks.mkdir(parents=True, exist_ok=True)
    for idx, nome in enumerate(nomes_patches):
        img = np.full((tamanho, tamanho, 3), (idx * 17) % 255, dtype=np.uint8)
        msk = np.zeros((tamanho, tamanho), dtype=np.uint8)
        if idx % 2 == 0:
            msk[: tamanho // 2, : tamanho // 2] = 255
        Image.fromarray(img, mode="RGB").save(dir_imgs / f"{nome}.png")
        Image.fromarray(msk, mode="L").save(dir_masks / f"{nome}.png")


def test_particionamento_disjunto_por_imagem_mae(tmp_path: Path) -> None:
    nomes_patches = [
        "healthy-01-roi1_p0000",
        "healthy-01-roi1_p0001",
        "healthy-01-roi1_p0002",
        "healthy-02-roi1_p0000",
        "healthy-02-roi1_p0001",
        "severe-01-roi1_p0000",
        "severe-02-roi1_p0000",
        "severe-02-roi1_p0001",
    ]
    dir_imgs = tmp_path / "imgs"
    dir_masks = tmp_path / "masks"
    _criar_patches_sinteticos(dir_imgs, dir_masks, nomes_patches)

    ds_p0 = HistologiaDataset(
        diretorio_imagens=str(dir_imgs),
        diretorio_mascaras=str(dir_masks),
        extensao_imagem=".png",
        num_particoes=2,
        indice_particao=0,
        seed_particao=42,
    )
    ds_p1 = HistologiaDataset(
        diretorio_imagens=str(dir_imgs),
        diretorio_mascaras=str(dir_masks),
        extensao_imagem=".png",
        num_particoes=2,
        indice_particao=1,
        seed_particao=42,
    )

    arquivos_p0 = {Path(img).stem for img, _ in ds_p0.amostras}
    arquivos_p1 = {Path(img).stem for img, _ in ds_p1.amostras}

    assert arquivos_p0.isdisjoint(arquivos_p1)
    assert arquivos_p0 | arquivos_p1 == set(nomes_patches)

    maes_p0 = {extrair_id_imagem_mae(nome) for nome in arquivos_p0}
    maes_p1 = {extrair_id_imagem_mae(nome) for nome in arquivos_p1}
    assert maes_p0.isdisjoint(maes_p1)
    assert len(maes_p0) == 2 and len(maes_p1) == 2

    # Determinismo com a mesma seed
    ds_p0_repetido = HistologiaDataset(
        diretorio_imagens=str(dir_imgs),
        diretorio_mascaras=str(dir_masks),
        extensao_imagem=".png",
        num_particoes=2,
        indice_particao=0,
        seed_particao=42,
    )
    assert [Path(img).stem for img, _ in ds_p0_repetido.amostras] == [
        Path(img).stem for img, _ in ds_p0.amostras
    ]

    with pytest.raises(ValueError):
        HistologiaDataset(str(dir_imgs), str(dir_masks), ".png", num_particoes=0)
    with pytest.raises(ValueError):
        HistologiaDataset(str(dir_imgs), str(dir_masks), ".png", num_particoes=2, indice_particao=2)


def test_criar_dataloaders_particiona_apenas_treino(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    splits_patches = {
        "train": ["imgA_p0000", "imgA_p0001", "imgB_p0000", "imgC_p0000", "imgD_p0000", "imgD_p0001"],
        "val": ["valA_p0000", "valB_p0000"],
        "test": ["testA_p0000", "testB_p0000", "testC_p0000"],
    }
    for split, nomes in splits_patches.items():
        _criar_patches_sinteticos(tmp_path / f"imgs_{split}", tmp_path / f"gt_{split}", nomes)

    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_IMGS_TRAIN", str(tmp_path / "imgs_train"))
    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_GT_TRAIN", str(tmp_path / "gt_train"))
    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_IMGS_VAL", str(tmp_path / "imgs_val"))
    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_GT_VAL", str(tmp_path / "gt_val"))
    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_IMGS_TEST", str(tmp_path / "imgs_test"))
    monkeypatch.setattr(dataloader_mod, "PATH_PATCHES_GT_TEST", str(tmp_path / "gt_test"))
    monkeypatch.setattr(dataloader_mod, "TAM_BATCH", 2)

    dl_tr_0, dl_val_0, dl_te_0 = dataloader_mod.criar_dataloaders(
        num_particoes=2, indice_particao=0, seed_particao=99
    )
    dl_tr_1, dl_val_1, dl_te_1 = dataloader_mod.criar_dataloaders(
        num_particoes=2, indice_particao=1, seed_particao=99
    )

    assert len(dl_tr_0.dataset) + len(dl_tr_1.dataset) == len(splits_patches["train"])
    assert len(dl_val_0.dataset) == len(dl_val_1.dataset) == len(splits_patches["val"])
    assert len(dl_te_0.dataset) == len(dl_te_1.dataset) == len(splits_patches["test"])


def test_ensemble_end_to_end_pesos_fixos_e_aprendiveis() -> None:
    x = torch.ones((2, 3, 8, 8), dtype=torch.float32)

    # Pesos fixos customizados (normalizados automaticamente para [0.75, 0.25])
    m1 = _ModeloConstante(2.0)
    m2 = _ModeloConstante(6.0)
    ens_fixo = EnsembleEndToEnd([m1, m2], pesos=[3.0, 1.0], pesos_aprendiveis=False)

    assert ens_fixo.logits_pesos is None
    assert pytest.approx(ens_fixo.obter_pesos_normalizados(), abs=1e-6) == [0.75, 0.25]
    saida_fixa = ens_fixo(x)
    assert saida_fixa.shape == (2, 1, 8, 8)
    assert torch.allclose(saida_fixa, torch.full((2, 1, 8, 8), 3.0))

    # Pesos aprendíveis + fluxo de gradientes com MuonAdamW
    sub1 = _ModeloLinear2D(escala=0.5)
    sub2 = _ModeloLinear2D(escala=-0.2)
    ens_apr = EnsembleEndToEnd([sub1, sub2], pesos_aprendiveis=True)

    assert ens_apr.logits_pesos is not None and ens_apr.logits_pesos.requires_grad
    assert pytest.approx(sum(ens_apr.obter_pesos_normalizados()), abs=1e-6) == 1.0

    otimizador = MuonAdamW(ens_apr.parameters(), lr=0.05, adamw_lr=0.05)
    alvo = torch.ones((2, 1, 8, 8), dtype=torch.float32)

    pesos_antes = ens_apr.obter_pesos_normalizados()
    w1_antes = sub1.proj.weight.detach().clone()
    w2_antes = sub2.proj.weight.detach().clone()

    otimizador.zero_grad()
    pred = ens_apr(x)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, alvo)
    loss.backward()

    assert ens_apr.logits_pesos.grad is not None
    assert sub1.proj.weight.grad is not None
    assert sub2.proj.weight.grad is not None

    otimizador.step()

    assert not torch.equal(sub1.proj.weight, w1_antes)
    assert not torch.equal(sub2.proj.weight, w2_antes)
    assert ens_apr.obter_pesos_normalizados() != pesos_antes
    assert pytest.approx(sum(ens_apr.obter_pesos_normalizados()), abs=1e-6) == 1.0


def test_ensemble_inferencia_todas_estrategias() -> None:
    x = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    m1 = _ModeloConstante(2.0)
    m2 = _ModeloConstante(0.0)

    p1 = torch.sigmoid(torch.tensor(2.0))
    p2 = torch.sigmoid(torch.tensor(0.0))

    esperados = {
        "media_logits": torch.tensor(1.0),
        "max_logits": torch.tensor(2.0),
        "media_probs": torch.logit(((p1 + p2) / 2.0).clamp(1e-6, 1.0 - 1e-6)),
        "max_probs": torch.logit(torch.maximum(p1, p2).clamp(1e-6, 1.0 - 1e-6)),
    }

    for estrategia in ESTRATEGIAS_AGREGACAO_ENSEMBLE:
        ens = EnsembleInferencia([m1, m2], estrategia_agregacao=estrategia)
        assert not ens.training
        assert all(not sub.training for sub in ens.modelos)

        saida = ens(x)
        assert not saida.requires_grad
        assert saida.shape == (2, 1, 4, 4)
        assert torch.allclose(saida, esperados[estrategia].expand_as(saida), atol=1e-5)

    with pytest.raises(ValueError):
        EnsembleInferencia([m1, m2], estrategia_agregacao="estrategia_inexistente")


def test_pegar_modelo_ensemble_e2e() -> None:
    modelo = pegar_modelo(
        "ENSEMBLE_E2E",
        arquiteturas=["UNET", "ATUNET"],
        pesos=[0.6, 0.4],
        pesos_aprendiveis=True,
    )
    assert isinstance(modelo, EnsembleEndToEnd)
    assert len(modelo.modelos) == 2
    assert pytest.approx(modelo.obter_pesos_normalizados(), abs=1e-6) == [0.6, 0.4]

    with pytest.raises(ValueError):
        pegar_modelo("ENSEMBLE_E2E", arquiteturas=["UNET"])


def test_avaliar_ensemble_checkpoints(tmp_path: Path) -> None:
    m1 = pegar_modelo("UNET").to(DEVICE)
    m2 = pegar_modelo("ATUNET").to(DEVICE)
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)

    ckpt1 = str(tmp_path / "unet_test.pt")
    ckpt2 = str(tmp_path / "atunet_test.pt")
    salvar_checkpoint(m1, opt1, epoca=1, caminho=ckpt1)
    salvar_checkpoint(m2, opt2, epoca=1, caminho=ckpt2)

    imagens = torch.randn((2, 3, TAM_PATCH, TAM_PATCH), dtype=torch.float32)
    mascaras = torch.randint(0, 2, (2, 1, TAM_PATCH, TAM_PATCH), dtype=torch.float32)
    dl_teste = DataLoader(TensorDataset(imagens, mascaras), batch_size=2)

    for estrategia in ESTRATEGIAS_AGREGACAO_ENSEMBLE:
        metricas = avaliar_ensemble_checkpoints(
            especificacoes_modelos=[("UNET", ckpt1), ("ATUNET", ckpt2)],
            dataloader_teste=dl_teste,
            estrategia_agregacao=estrategia,
        )
        assert set(metricas.keys()) == {"classe_0", "classe_1", "mdice", "miou"}
        assert 0.0 <= metricas["mdice"] <= 1.0
        assert 0.0 <= metricas["miou"] <= 1.0


def test_salvar_resultado_completo_com_metadados_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caminho_csv = tmp_path / "tabela_completa.csv"
    monkeypatch.setattr(tabelas_mod, "PATH_RESULTS_TABELAS_TABELA_COMPLETA", str(caminho_csv))

    metricas_dummy = {
        "classe_0": {"dice": 0.9, "iou": 0.82, "precision": 0.91, "recall": 0.89},
        "classe_1": {"dice": 0.8, "iou": 0.67, "precision": 0.83, "recall": 0.77},
        "mdice": 0.85,
        "miou": 0.745,
    }

    tabelas_mod.salvar_resultado_completo(
        nome_modelo="E2E_UNET+ATUNET_aprendivel_FOCALDICE_ADAMW_COSSENO_PADRAO_lr0.001_augTrue_seed42",
        num_epocas=10,
        lr=0.001,
        early_stopper="PADRAO",
        otimizador="ADAMW",
        scheduler="COSSENO",
        loss="FOCALDICE",
        usou_aug=True,
        melhor_epoca=8,
        melhor_mdice=0.84,
        metricas_avaliacao=metricas_dummy,
        seed=42,
        metadados_extras={"tipo_ensemble": "end_to_end", "peso_modelo_0": 0.55, "peso_modelo_1": 0.45},
    )

    df = pd.read_csv(caminho_csv)
    assert len(df) == 1
    assert df.loc[0, "tipo_ensemble"] == "end_to_end"
    assert pytest.approx(df.loc[0, "peso_modelo_0"]) == 0.55
    assert pytest.approx(df.loc[0, "peso_modelo_1"]) == 0.45
