""" Factory de modelos """

from typing import Any, Dict, Optional, Sequence

import torch.nn as nn

from src.modelos.attention_unet import AttentionUNet
from src.modelos.ensemble import EnsembleEndToEnd
from src.modelos.unet import UNet
from src.modelos.ukan import UKAN
from src.modelos.segformer import SegFormer


def pegar_modelo(nome: str, **kwargs: Any) -> nn.Module:
    """
    Modelos possíveis:

    - "ATUNET": Attention U-Net definida em src_tema/src/modelos/attention_unet.py
    - "UNET": U-Net padrão definida em src_tema/src/modelos/unet.py
    - "UKAN": U-KAN definida em src_tema/src/modelos/ukan.py
    - "SEGFORMER": Segformer definida em src_tema/src/modelos/segformer.py
    - "ENSEMBLE_E2E": EnsembleEndToEnd definido em src_tema/src/modelos/ensemble.py
      (requer `arquiteturas: Sequence[str]`, aceitando `pesos` e `pesos_aprendiveis`)

    Levanta ValueError caso o nome informado seja diferente de
    um listado ou se os parâmetros obrigatórios estiverem ausentes.
    """
    nome_upper = nome.upper().strip()

    if nome_upper == "ATUNET":
        return AttentionUNet(**kwargs)
    elif nome_upper == "UNET":
        return UNet(**kwargs)
    elif nome_upper == "UKAN":
        return UKAN(**kwargs)
    elif nome_upper == "SEGFORMER":
        return SegFormer(**kwargs)
    elif nome_upper == "ENSEMBLE_E2E":
        arquiteturas: Optional[Sequence[str]] = kwargs.get("arquiteturas")
        if arquiteturas is None or len(arquiteturas) < 2:
            raise ValueError(
                "Para 'ENSEMBLE_E2E', é obrigatório fornecer 'arquiteturas' com pelo menos 2 modelos."
            )

        submodelos_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = kwargs.get("submodelos_kwargs")
        if submodelos_kwargs is not None:
            if len(submodelos_kwargs) != len(arquiteturas):
                raise ValueError(
                    "submodelos_kwargs deve possuir o mesmo comprimento de arquiteturas."
                )
            modelos = [
                pegar_modelo(arq, **(sub_kw or {}))
                for arq, sub_kw in zip(arquiteturas, submodelos_kwargs)
            ]
        else:
            modelos = [pegar_modelo(arq) for arq in arquiteturas]

        pesos: Optional[Sequence[float]] = kwargs.get("pesos", None)
        pesos_aprendiveis: bool = bool(kwargs.get("pesos_aprendiveis", False))
        return EnsembleEndToEnd(
            modelos=modelos,
            pesos=pesos,
            pesos_aprendiveis=pesos_aprendiveis,
        )
    else:
        raise ValueError(f"Arquitetura desconhecida: '{nome}'")
