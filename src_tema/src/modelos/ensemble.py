"""
Módulos de Ensemble para segmentação semântica binária (treinamento conjunto
end-to-end e agregação na inferência).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from configs.basicas import (
    ESTRATEGIAS_AGREGACAO_ENSEMBLE,
    ESTRATEGIA_AGREGACAO_PADRAO,
)


class EnsembleEndToEnd(nn.Module):
    """
    Wrapper unificado para treinamento conjunto (end-to-end) de 2 ou mais
    modelos de segmentação binária.

    Todos os submodelos recebem o mesmo tensor de entrada simultaneamente,
    geram seus logits `[B, 1, H, W]` e têm suas saídas combinadas por uma
    média ponderada normalizada (com pesos fixos ou aprendíveis via softmax).
    """

    def __init__(
        self,
        modelos: Sequence[nn.Module],
        pesos: Optional[Sequence[float]] = None,
        pesos_aprendiveis: bool = False,
    ) -> None:
        super().__init__()
        if not modelos or len(modelos) < 2:
            raise ValueError("EnsembleEndToEnd requer pelo menos 2 modelos.")
        if not all(isinstance(modelo, nn.Module) for modelo in modelos):
            raise TypeError("Todos os elementos de 'modelos' devem ser instâncias de nn.Module.")

        self.modelos = nn.ModuleList(list(modelos))
        self.pesos_aprendiveis = bool(pesos_aprendiveis)
        num_modelos = len(self.modelos)

        if pesos is None:
            pesos_normalizados = torch.full((num_modelos,), 1.0 / num_modelos, dtype=torch.float32)
        else:
            if len(pesos) != num_modelos:
                raise ValueError(
                    f"Quantidade de pesos ({len(pesos)}) difere da quantidade de modelos ({num_modelos})."
                )
            pesos_tensor = torch.tensor(list(pesos), dtype=torch.float32)
            if not torch.isfinite(pesos_tensor).all() or (pesos_tensor < 0).any() or pesos_tensor.sum() <= 0:
                raise ValueError("Os pesos devem ser finitos, não-negativos e possuir soma positiva.")
            pesos_normalizados = pesos_tensor / pesos_tensor.sum()

        if self.pesos_aprendiveis:
            if pesos is None:
                logits_iniciais = torch.zeros(num_modelos, dtype=torch.float32)
            else:
                logits_iniciais = torch.log(pesos_normalizados.clamp(min=1e-8))
            self.logits_pesos: Optional[nn.Parameter] = nn.Parameter(logits_iniciais)
            self.register_buffer("pesos_fixos", None)
        else:
            self.logits_pesos = None
            self.register_buffer("pesos_fixos", pesos_normalizados)

    def _tensor_pesos_normalizados(self) -> torch.Tensor:
        """Retorna o tensor 1D `[M]` de pesos normalizados que somam 1.0."""
        if self.pesos_aprendiveis:
            assert self.logits_pesos is not None
            return torch.softmax(self.logits_pesos, dim=0)
        assert self.pesos_fixos is not None
        return self.pesos_fixos

    def obter_pesos_normalizados(self) -> List[float]:
        """Retorna a lista Python com os pesos normalizados atuais de cada submodelo."""
        pesos_atuais = self._tensor_pesos_normalizados().detach().cpu()
        return [float(peso) for peso in pesos_atuais.tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa todos os submodelos em paralelo lógico sobre `x` e combina
        os logits empilhados `[M, B, 1, H, W]` em `[B, 1, H, W]`.
        """
        saidas = [modelo(x) for modelo in self.modelos]
        forma_referencia = saidas[0].shape
        if any(saida.shape != forma_referencia for saida in saidas):
            raise ValueError(
                "Todos os submodelos devem produzir tensores de saída com o mesmo formato."
            )

        logits_empilhados = torch.stack(saidas, dim=0)
        pesos = self._tensor_pesos_normalizados()
        pesos_expandidos = pesos.view(-1, *([1] * (logits_empilhados.ndim - 1)))
        return (logits_empilhados * pesos_expandidos).sum(dim=0)


class EnsembleInferencia(nn.Module):
    """
    Wrapper de inferência para combinar saídas de 2 ou mais modelos treinados
    independentemente (ensemble heterogêneo ou bagging disjunto).
    """

    def __init__(
        self,
        modelos: Sequence[nn.Module],
        estrategia_agregacao: str = ESTRATEGIA_AGREGACAO_PADRAO,
    ) -> None:
        super().__init__()
        if not modelos or len(modelos) < 2:
            raise ValueError("EnsembleInferencia requer pelo menos 2 modelos.")
        if not all(isinstance(modelo, nn.Module) for modelo in modelos):
            raise TypeError("Todos os elementos de 'modelos' devem ser instâncias de nn.Module.")

        estrategia_normalizada = estrategia_agregacao.lower().strip()
        if estrategia_normalizada not in ESTRATEGIAS_AGREGACAO_ENSEMBLE:
            raise ValueError(
                f"Estratégia de agregação '{estrategia_agregacao}' inválida. "
                f"Opções válidas: {ESTRATEGIAS_AGREGACAO_ENSEMBLE}."
            )

        self.modelos = nn.ModuleList(list(modelos))
        self.estrategia_agregacao = estrategia_normalizada
        self.eval()

    def train(self, mode: bool = True) -> "EnsembleInferencia":
        """Mantém o ensemble de inferência e todos os seus submodelos sempre em `.eval()`."""
        super().train(False)
        for modelo in self.modelos:
            modelo.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Agrega os logits/probabilidades dos submodelos e retorna um tensor de
        logits `[B, 1, H, W]` compatível com `avaliar_modelo` e `logits_para_mascara`.
        """
        saidas = [modelo(x) for modelo in self.modelos]
        forma_referencia = saidas[0].shape
        if any(saida.shape != forma_referencia for saida in saidas):
            raise ValueError(
                "Todos os submodelos devem produzir tensores de saída com o mesmo formato."
            )

        logits_empilhados = torch.stack(saidas, dim=0)

        if self.estrategia_agregacao == "media_logits":
            return logits_empilhados.mean(dim=0)
        if self.estrategia_agregacao == "max_logits":
            return logits_empilhados.max(dim=0).values
        if self.estrategia_agregacao == "media_probs":
            probs = torch.sigmoid(logits_empilhados).mean(dim=0)
            return torch.logit(probs.clamp(min=1e-6, max=1.0 - 1e-6))
        if self.estrategia_agregacao == "max_probs":
            probs = torch.sigmoid(logits_empilhados).max(dim=0).values
            return torch.logit(probs.clamp(min=1e-6, max=1.0 - 1e-6))

        raise ValueError(f"Estratégia de agregação desconhecida: '{self.estrategia_agregacao}'.")
    