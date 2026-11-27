"""
Treina e avalia as três estratégias de ensemble nas seeds 42, 125 e 2026:

1. Ensemble Heterogêneo (Inferência):
   Treina U-KAN e SegFormer de forma independente e combina suas predições
   na inferência (ex.: média das probabilidades).
2. Ensemble Homogêneo com Particionamento (Bagging Disjunto):
   Divide o dataset de treino em 2 partições mutuamente exclusivas agrupadas
   por imagem-mãe, treina 2 instâncias de U-KAN separadamente e agrega suas
   predições na inferência.
3. Ensemble End-to-End Ponderado (Treino Conjunto):
   Treina U-KAN e SegFormer conjuntamente em um único fluxo onde ambos
   recebem a mesma imagem e têm seus logits combinados por média ponderada
   (com pesos fixos ou aprendíveis) antes do cálculo da loss Focal+Dice.

Uso (a partir da raiz src_tema):
    python -m scripts.treinar_ensembles
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Garante execução tanto via `python -m scripts.treinar_ensembles` quanto diretamente
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import pandas as pd

import configs.basicas as cfg_basicas
import src.schedulerslr.scheduler_factory as scheduler_factory_mod
from configs.basicas import (
    ESTRATEGIAS_AGREGACAO_ENSEMBLE,
    ESTRATEGIA_AGREGACAO_PADRAO,
    TAXA_APRENDIZADO,
)
from src.avaliacao.metricas import MetricasBinarias, agregar_repeticoes
from src.utils.paths import PATH_RESULTS_TABELAS
from src.utils.rodar_experimentos import (
    rodar_ensemble_bagging_disjunto,
    rodar_ensemble_end_to_end,
    rodar_ensemble_heterogeneo,
)

# Configurações fixas do experimento
SEEDS: Sequence[int] = (42, 125, 2026)
NUM_EPOCAS_PADRAO: int = 100
NOME_LOSS: str = "FOCALDICE"
NOME_OTIMIZADOR: str = "ADAMW"
NOME_SCHEDULER: str = "WARMUP_COSSENO"
NOME_EARLY_STOPPER: str = "PADRAO"

ARQUITETURAS_HETEROGENEO: Sequence[str] = ("UKAN", "SEGFORMER")
ARQUITETURA_BAGGING: str = "UKAN"
NUM_PARTICOES_BAGGING: int = 2
ARQUITETURAS_E2E: Sequence[str] = ("UKAN", "SEGFORMER")

NOMES_EXIBICAO: Dict[str, str] = {
    "heterogeneo": "1. Ensemble Heterogêneo (U-KAN + SegFormer)",
    "bagging": "2. Ensemble Homogêneo - Bagging Disjunto (2x U-KAN)",
    "end_to_end": "3. Ensemble End-to-End Ponderado (U-KAN + SegFormer)",
}


def sincronizar_num_epocas(num_epocas: int) -> None:
    """
    Atualiza NUM_EPOCAS em `configs.basicas` e em `scheduler_factory` para que
    `WARMUP_COSSENO` utilize `T_max = num_epocas` (e `T_warmup = 0.1 * num_epocas`).
    """
    cfg_basicas.NUM_EPOCAS = num_epocas
    scheduler_factory_mod.NUM_EPOCAS = num_epocas


def executar_experimentos_ensembles(
    tipos_ensemble: Sequence[str],
    seeds: Sequence[int] = SEEDS,
    num_epocas: int = NUM_EPOCAS_PADRAO,
    lr: float = TAXA_APRENDIZADO,
    aplicar_aug: bool = True,
    estrategia_agregacao: str = ESTRATEGIA_AGREGACAO_PADRAO,
    modo_pesos_e2e: str = "aprendivel",
    num_workers: int = 0,
    fracao_dados: float = 1.0,
    reutilizar_checkpoints: bool = True,
) -> Dict[str, List[MetricasBinarias]]:
    """Executa cada estratégia de ensemble solicitada para todas as seeds."""
    sincronizar_num_epocas(num_epocas)
    resultados_por_tipo: Dict[str, List[MetricasBinarias]] = {
        tipo: [] for tipo in tipos_ensemble
    }

    for tipo in tipos_ensemble:
        print("\n" + "#" * 80)
        print(f"INICIANDO: {NOMES_EXIBICAO[tipo]}")
        print("#" * 80)

        for seed in seeds:
            print(f"\n>>> [{tipo.upper()}] Executando Seed {seed}...")

            if tipo == "heterogeneo":
                metricas = rodar_ensemble_heterogeneo(
                    arquiteturas=ARQUITETURAS_HETEROGENEO,
                    nome_loss=NOME_LOSS,
                    nome_otimizador=NOME_OTIMIZADOR,
                    nome_scheduler=NOME_SCHEDULER,
                    nome_early_stopper=NOME_EARLY_STOPPER,
                    lr=lr,
                    num_epocas=num_epocas,
                    aplicar_aug=aplicar_aug,
                    num_workers=num_workers,
                    seed=seed,
                    fracao_dados=fracao_dados,
                    estrategia_agregacao=estrategia_agregacao,
                    reutilizar_checkpoints=reutilizar_checkpoints,
                )

            elif tipo == "bagging":
                metricas = rodar_ensemble_bagging_disjunto(
                    nome_arquitetura=ARQUITETURA_BAGGING,
                    nome_loss=NOME_LOSS,
                    nome_otimizador=NOME_OTIMIZADOR,
                    nome_scheduler=NOME_SCHEDULER,
                    nome_early_stopper=NOME_EARLY_STOPPER,
                    lr=lr,
                    num_epocas=num_epocas,
                    aplicar_aug=aplicar_aug,
                    num_workers=num_workers,
                    seed=seed,
                    fracao_dados=fracao_dados,
                    num_particoes=NUM_PARTICOES_BAGGING,
                    seed_particao=seed,
                    estrategia_agregacao=estrategia_agregacao,
                    reutilizar_checkpoints=reutilizar_checkpoints,
                )

            elif tipo == "end_to_end":
                metricas = rodar_ensemble_end_to_end(
                    arquiteturas=ARQUITETURAS_E2E,
                    nome_loss=NOME_LOSS,
                    nome_otimizador=NOME_OTIMIZADOR,
                    nome_scheduler=NOME_SCHEDULER,
                    nome_early_stopper=NOME_EARLY_STOPPER,
                    lr=lr,
                    num_epocas=num_epocas,
                    aplicar_aug=aplicar_aug,
                    num_workers=num_workers,
                    seed=seed,
                    fracao_dados=fracao_dados,
                    modo_pesos=modo_pesos_e2e,
                )

            else:
                raise ValueError(f"Tipo de ensemble desconhecido: '{tipo}'.")

            if not metricas or "mdice" not in metricas:
                raise RuntimeError(
                    f"O treinamento falhou ou divergiu para '{tipo}' na seed {seed}."
                )

            resultados_por_tipo[tipo].append(metricas)  # type: ignore[arg-type]

    return resultados_por_tipo


def imprimir_e_salvar_resumo(
    resultados_por_tipo: Dict[str, List[MetricasBinarias]],
    seeds: Sequence[int],
) -> None:
    """
    Calcula média e desvio padrão do mDice de teste entre as seeds para cada
    tipo de ensemble, imprime o relatório final e salva um CSV resumido.
    """
    print("\n" + "=" * 80)
    print("RESUMO FINAL - mDice NO CONJUNTO DE TESTE (Seeds: " + ", ".join(map(str, seeds)) + ")")
    print("=" * 80)

    linhas_resumo = []

    for tipo, lista_metricas in resultados_por_tipo.items():
        titulo = NOMES_EXIBICAO[tipo]
        estatisticas = agregar_repeticoes(lista_metricas)
        mdice_stats = estatisticas["mdice"]  # type: ignore[index]
        miou_stats = estatisticas["miou"]  # type: ignore[index]

        media_mdice = float(mdice_stats["media"])  # type: ignore[index]
        std_mdice = float(mdice_stats["desvio_padrao"])  # type: ignore[index]
        media_miou = float(miou_stats["media"])  # type: ignore[index]
        std_miou = float(miou_stats["desvio_padrao"])  # type: ignore[index]

        detalhes_seeds = " | ".join(
            f"seed {s}: {m['mdice']:.4f}" for s, m in zip(seeds, lista_metricas)
        )

        print(f"\n{titulo}")
        print(f"  Resultados por seed : {detalhes_seeds}")
        print(f"  mDice Teste         : {media_mdice:.4f} ± {std_mdice:.4f} (média = {media_mdice:.6f}, desvio padrão = {std_mdice:.6f})")
        print(f"  mIoU Teste          : {media_miou:.4f} ± {std_miou:.4f}")

        linha = {
            "tipo_ensemble": tipo,
            "descricao": titulo,
            "mdice_teste_media": media_mdice,
            "mdice_teste_desvio_padrao": std_mdice,
            "miou_teste_media": media_miou,
            "miou_teste_desvio_padrao": std_miou,
        }
        for s, m in zip(seeds, lista_metricas):
            linha[f"mdice_seed_{s}"] = m["mdice"]
        linhas_resumo.append(linha)

    print("\n" + "=" * 80)

    os.makedirs(PATH_RESULTS_TABELAS, exist_ok=True)
    caminho_resumo = os.path.join(PATH_RESULTS_TABELAS, "resumo_ensembles.csv")
    pd.DataFrame(linhas_resumo).to_csv(caminho_resumo, index=False)
    print(f"[INFO] Resumo consolidado salvo em: {caminho_resumo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia ensembles (Heterogêneo, Bagging Disjunto e End-to-End) nas seeds 42, 125 e 2026."
    )
    parser.add_argument(
        "--ensembles",
        nargs="+",
        choices=("heterogeneo", "bagging", "end_to_end"),
        default=("heterogeneo", "bagging", "end_to_end"),
        help="Tipos de ensemble a executar (padrão: todos os 3).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TAXA_APRENDIZADO,
        help=f"Taxa de aprendizado para o AdamW (padrão: {TAXA_APRENDIZADO}).",
    )
    parser.add_argument(
        "--num-epocas",
        type=int,
        default=NUM_EPOCAS_PADRAO,
        help=f"Número máximo de épocas por treino (padrão: {NUM_EPOCAS_PADRAO}).",
    )
    parser.add_argument(
        "--estrategia-agregacao",
        type=str,
        choices=ESTRATEGIAS_AGREGACAO_ENSEMBLE,
        default=ESTRATEGIA_AGREGACAO_PADRAO,
        help=f"Operação de agregação na inferência para Heterogêneo e Bagging (padrão: {ESTRATEGIA_AGREGACAO_PADRAO}).",
    )
    parser.add_argument(
        "--modo-pesos",
        type=str,
        choices=("fixo", "aprendivel"),
        default="aprendivel",
        help="Modo dos pesos na média ponderada do Ensemble End-to-End (padrão: aprendivel).",
    )
    parser.add_argument(
        "--sem-aug",
        action="store_true",
        help="Desativa data augmentation no conjunto de treino.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de workers dos DataLoaders (padrão: 0).",
    )
    parser.add_argument(
        "--fracao-dados",
        type=float,
        default=1.0,
        help="Fração do dataset utilizada (padrão: 1.0).",
    )
    parser.add_argument(
        "--forcar-retreino",
        action="store_true",
        help="Ignora checkpoints já salvos em results/modelos e treina tudo do zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resultados = executar_experimentos_ensembles(
        tipos_ensemble=args.ensembles,
        seeds=SEEDS,
        num_epocas=args.num_epocas,
        lr=args.lr,
        aplicar_aug=not args.sem_aug,
        estrategia_agregacao=args.estrategia_agregacao,
        modo_pesos_e2e=args.modo_pesos,
        num_workers=args.num_workers,
        fracao_dados=args.fracao_dados,
        reutilizar_checkpoints=not args.forcar_retreino,
    )
    imprimir_e_salvar_resumo(resultados, seeds=SEEDS)


if __name__ == "__main__":
    main()
