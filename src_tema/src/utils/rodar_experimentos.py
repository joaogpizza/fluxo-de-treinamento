"""
Módulo para automatizar e orquestrar o fluxo completo de experimentos de segmentação,
incluindo suporte a ensembles heterogêneos, bagging disjunto e treinamento end-to-end.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.basicas import (
    DEVICE,
    ESTRATEGIAS_AGREGACAO_ENSEMBLE,
    ESTRATEGIA_AGREGACAO_PADRAO,
    MODO_PESOS_ENSEMBLE_PADRAO,
    NUM_EPOCAS,
    SEED_SPLIT,
    TAXA_APRENDIZADO,
)
from src.avaliacao.avaliador import avaliar_ensemble_checkpoints, avaliar_modelo
from src.data.dataloader import criar_dataloaders
from src.modelos.modelo_factory import pegar_modelo
from src.treino.treinador import treinar_modelo
from src.utils.checkpoints import carregar_checkpoint
from src.utils.paths import PATH_RESULTS_MODELOS
from src.utils.seed import fixar_seed
from src.utils.tabelas import salvar_resultado_completo, salvar_tabela_experimentos


def _formatar_nome_experimento(
    nome_arquitetura: str,
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float,
    aplicar_aug: bool,
    seed: int,
    sufixo_nome: str = "",
) -> str:
    """Constrói o identificador padronizado de um experimento."""
    nome_base = (
        f"{nome_arquitetura}_{nome_loss}_{nome_otimizador}_{nome_scheduler}_"
        f"{nome_early_stopper}_lr{lr}_aug{aplicar_aug}_seed{seed}"
    )
    if not sufixo_nome:
        return nome_base
    sufixo_formatado = sufixo_nome if sufixo_nome.startswith("_") else f"_{sufixo_nome}"
    return f"{nome_base}{sufixo_formatado}"


def rodar_experimento_completo(
    nome_arquitetura: str,
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float = TAXA_APRENDIZADO,
    num_epocas: int = NUM_EPOCAS,
    aplicar_aug: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    fracao_dados: float = 1.0,
    num_particoes: int = 1,
    indice_particao: int = 0,
    seed_particao: int = SEED_SPLIT,
    sufixo_nome: str = "",
    modelo_kwargs: Optional[Dict[str, Any]] = None,
    metadados_extras: Optional[Dict[str, Any]] = None,
    nome_modelo: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """
    Roda o pipeline completo de treinamento e avaliação de um modelo ou ensemble.

    Args:
        nome_arquitetura (str): Arquitetura (ex: "UNET", "ATUNET", "UKAN", "ENSEMBLE_E2E").
        nome_loss (str): Função de custo (ex: "DICE", "FOCAL", "FOCALDICE").
        nome_otimizador (str): Otimizador (ex: "ADAMW", "MUON").
        nome_scheduler (str): Agendador de Learning Rate (ex: "COSSENO").
        nome_early_stopper (str): Critério de parada (ex: "PADRAO", "PQ").
        lr (float, optional): Taxa de aprendizado. Default = TAXA_APRENDIZADO.
        num_epocas (int, optional): Número de épocas. Default = NUM_EPOCAS.
        aplicar_aug (bool, optional): Se aplica Data Augmentation. Default = False.
        num_workers (int, optional): Workers para o DataLoader. Default = 0.
        seed (int, optional): Semente aleatória. Default = 42.
        fracao_dados (float, optional): Fração do dataset a utilizar. Default = 1.0.
        num_particoes (int, optional): Número de partições disjuntas no treino. Default = 1.
        indice_particao (int, optional): Índice da partição selecionada. Default = 0.
        seed_particao (int, optional): Semente para particionamento por imagem-mãe. Default = SEED_SPLIT.
        sufixo_nome (str, optional): Sufixo opcional anexado ao nome do modelo. Default = "".
        modelo_kwargs (dict, optional): Argumentos extras repassados a `pegar_modelo`.
        metadados_extras (dict, optional): Colunas extras salvas em `tabela_completa.csv`.
        nome_modelo (str, optional): Sobrescreve o nome padronizado do experimento quando informado.
        **kwargs: Argumentos extras repassados para a loss, otimizador, scheduler, etc.

    Returns:
        dict: Métricas do modelo avaliado no conjunto de teste.
    """
    sufixo_efetivo = sufixo_nome
    if not sufixo_efetivo and num_particoes > 1:
        sufixo_efetivo = f"part{indice_particao}de{num_particoes}"

    nome_experimento = nome_modelo or _formatar_nome_experimento(
        nome_arquitetura=nome_arquitetura,
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        lr=lr,
        aplicar_aug=aplicar_aug,
        seed=seed,
        sufixo_nome=sufixo_efetivo,
    )

    print(f"{'='*60}")
    print(f"Iniciando Experimento: {nome_experimento}")
    print(f"{'='*60}")

    fixar_seed(seed)
    print(f"[INFO] Seed fixada em {seed}.")

    print("[INFO] Criando DataLoaders (Treino, Validação, Teste)...")
    dl_treino, dl_val, dl_teste = criar_dataloaders(
        num_workers=num_workers,
        aplicar_aug=aplicar_aug,
        fracao_dados=fracao_dados,
        num_particoes=num_particoes,
        indice_particao=indice_particao,
        seed_particao=seed_particao,
    )

    print("[INFO] Iniciando rotina de treinamento...")
    melhor_mdice, melhor_epoca, divergiu, tempo_treino = treinar_modelo(
        nome_modelo=nome_experimento,
        num_epocas=num_epocas,
        lr=lr,
        nome_arquitetura=nome_arquitetura,
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        dataloader_treino=dl_treino,
        dataloader_val=dl_val,
        modelo_kwargs=modelo_kwargs or {},
        **kwargs,
    )

    if not divergiu:
        print("\n[INFO] Carregando o melhor checkpoint para avaliação no conjunto de teste...")
        modelo = pegar_modelo(nome_arquitetura, **(modelo_kwargs or {})).to(DEVICE)
        caminho_checkpoint = os.path.join(PATH_RESULTS_MODELOS, f"{nome_experimento}.pt")

        carregar_checkpoint(modelo, caminho=caminho_checkpoint)

        print("[INFO] Avaliando o modelo (Teste)...")
        metricas_avaliacao = avaliar_modelo(modelo, dl_teste)
        print(f"[RESULTADOS TESTE] mDice: {metricas_avaliacao['mdice']:.4f} | mIoU: {metricas_avaliacao['miou']:.4f}")

        metadados_finais: Dict[str, Any] = dict(metadados_extras) if metadados_extras else {}
        if num_particoes > 1:
            metadados_finais.setdefault("num_particoes", num_particoes)
            metadados_finais.setdefault("indice_particao", indice_particao)
            metadados_finais.setdefault("seed_particao", seed_particao)

        if hasattr(modelo, "obter_pesos_normalizados") and callable(modelo.obter_pesos_normalizados):
            pesos_finais = modelo.obter_pesos_normalizados()
            metadados_finais["pesos_ensemble"] = json.dumps([round(p, 6) for p in pesos_finais])
            for idx_peso, valor_peso in enumerate(pesos_finais):
                metadados_finais[f"peso_modelo_{idx_peso}"] = float(valor_peso)

        print("\n[INFO] Salvando resultados nas tabelas (completa e de experimentos)...")
        salvar_resultado_completo(
            nome_modelo=nome_experimento,
            num_epocas=num_epocas,
            lr=lr,
            early_stopper=nome_early_stopper,
            otimizador=nome_otimizador,
            scheduler=nome_scheduler,
            loss=nome_loss,
            usou_aug=aplicar_aug,
            melhor_epoca=melhor_epoca,
            melhor_mdice=melhor_mdice,
            metricas_avaliacao=metricas_avaliacao,
            seed=seed,
            metadados_extras=metadados_finais or None,
        )

        salvar_tabela_experimentos(
            nome_modelo=nome_experimento,
            num_epocas=num_epocas,
            lr=lr,
            early_stopper=nome_early_stopper,
            otimizador=nome_otimizador,
            scheduler=nome_scheduler,
            loss=nome_loss,
            usou_aug=aplicar_aug,
            melhor_mdice=melhor_mdice,
            divergiu=divergiu,
            tempo_treino=tempo_treino,
        )

        print(f"\n[SUCESSO] Experimento '{nome_experimento}' finalizado!")
        return metricas_avaliacao
    else:
        salvar_tabela_experimentos(
            nome_modelo=nome_experimento,
            num_epocas=num_epocas,
            lr=lr,
            early_stopper=nome_early_stopper,
            otimizador=nome_otimizador,
            scheduler=nome_scheduler,
            loss=nome_loss,
            usou_aug=aplicar_aug,
            melhor_mdice=0.0,
            divergiu=divergiu,
            tempo_treino=tempo_treino,
        )
        return {}


def rodar_ensemble_heterogeneo(
    arquiteturas: Sequence[str],
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float = TAXA_APRENDIZADO,
    num_epocas: int = NUM_EPOCAS,
    aplicar_aug: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    fracao_dados: float = 1.0,
    estrategia_agregacao: str = ESTRATEGIA_AGREGACAO_PADRAO,
    configs_modelos: Optional[Sequence[Dict[str, Any]]] = None,
    reutilizar_checkpoints: bool = True,
    sufixo_nome: str = "",
    metadados_extras: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """
    Orquestra a Estratégia 1 (Ensemble Heterogêneo):
    treina modelos de arquiteturas distintas de forma independente (reutilizando
    checkpoints já existentes quando disponíveis), avalia o conjunto no teste
    via `avaliar_ensemble_checkpoints` e salva resultados nas tabelas CSV.
    """
    if not arquiteturas or len(arquiteturas) < 2:
        raise ValueError("rodar_ensemble_heterogeneo requer pelo menos 2 arquiteturas.")

    estrategia_norm = estrategia_agregacao.lower().strip()
    if estrategia_norm not in ESTRATEGIAS_AGREGACAO_ENSEMBLE:
        raise ValueError(
            f"Estratégia de agregação '{estrategia_agregacao}' inválida. "
            f"Opções válidas: {ESTRATEGIAS_AGREGACAO_ENSEMBLE}."
        )

    inicio_total = time.time()
    especificacoes_modelos: List[Tuple[str, str]] = []
    nomes_base_modelos: List[str] = []

    for idx, arq in enumerate(arquiteturas):
        cfg = dict(configs_modelos[idx]) if (configs_modelos and idx < len(configs_modelos)) else {}
        arq_loss = cfg.get("nome_loss", nome_loss)
        arq_opt = cfg.get("nome_otimizador", nome_otimizador)
        arq_sched = cfg.get("nome_scheduler", nome_scheduler)
        arq_es = cfg.get("nome_early_stopper", nome_early_stopper)
        arq_lr = cfg.get("lr", lr)
        arq_epocas = cfg.get("num_epocas", num_epocas)
        arq_aug = cfg.get("aplicar_aug", aplicar_aug)
        arq_seed = cfg.get("seed", seed)
        arq_sufixo = cfg.get("sufixo_nome", "")
        arq_modelo_kwargs = cfg.get("modelo_kwargs", None)

        nome_submodelo = _formatar_nome_experimento(
            nome_arquitetura=arq,
            nome_loss=arq_loss,
            nome_otimizador=arq_opt,
            nome_scheduler=arq_sched,
            nome_early_stopper=arq_es,
            lr=arq_lr,
            aplicar_aug=arq_aug,
            seed=arq_seed,
            sufixo_nome=arq_sufixo,
        )
        caminho_ckpt = os.path.join(PATH_RESULTS_MODELOS, f"{nome_submodelo}.pt")

        if reutilizar_checkpoints and os.path.exists(caminho_ckpt):
            print(f"[CACHE] Reutilizando checkpoint existente: {caminho_ckpt}")
        else:
            metricas_sub = rodar_experimento_completo(
                nome_arquitetura=arq,
                nome_loss=arq_loss,
                nome_otimizador=arq_opt,
                nome_scheduler=arq_sched,
                nome_early_stopper=arq_es,
                lr=arq_lr,
                num_epocas=arq_epocas,
                aplicar_aug=arq_aug,
                num_workers=num_workers,
                seed=arq_seed,
                fracao_dados=fracao_dados,
                sufixo_nome=arq_sufixo,
                modelo_kwargs=arq_modelo_kwargs,
                **kwargs,
            )
            if not metricas_sub or not os.path.exists(caminho_ckpt):
                raise RuntimeError(
                    f"Falha ao treinar o submodelo '{nome_submodelo}' para o ensemble heterogêneo."
                )

        especificacoes_modelos.append((arq, caminho_ckpt))
        nomes_base_modelos.append(nome_submodelo)

    _, _, dl_teste = criar_dataloaders(
        num_workers=num_workers,
        aplicar_aug=False,
        fracao_dados=fracao_dados,
    )

    metricas_avaliacao = avaliar_ensemble_checkpoints(
        especificacoes_modelos=especificacoes_modelos,
        dataloader_teste=dl_teste,
        estrategia_agregacao=estrategia_norm,
    )

    arqs_tag = "+".join(arq.upper() for arq in arquiteturas)
    nome_ensemble = _formatar_nome_experimento(
        nome_arquitetura=f"ENS_HET_{arqs_tag}_{estrategia_norm}",
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        lr=lr,
        aplicar_aug=aplicar_aug,
        seed=seed,
        sufixo_nome=sufixo_nome,
    )

    metadados_finais: Dict[str, Any] = dict(metadados_extras) if metadados_extras else {}
    metadados_finais.update({
        "tipo_ensemble": "heterogeneo",
        "arquiteturas": arqs_tag,
        "estrategia_agregacao": estrategia_norm,
        "modelos_base": json.dumps(nomes_base_modelos),
    })

    tempo_total = time.time() - inicio_total
    salvar_resultado_completo(
        nome_modelo=nome_ensemble,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_epoca=num_epocas,
        melhor_mdice=metricas_avaliacao["mdice"],
        metricas_avaliacao=metricas_avaliacao,
        seed=seed,
        metadados_extras=metadados_finais,
    )

    salvar_tabela_experimentos(
        nome_modelo=nome_ensemble,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_mdice=metricas_avaliacao["mdice"],
        divergiu=False,
        tempo_treino=tempo_total,
    )

    print(
        f"\n[SUCESSO] Ensemble Heterogêneo '{nome_ensemble}' finalizado | "
        f"mDice: {metricas_avaliacao['mdice']:.4f} | mIoU: {metricas_avaliacao['miou']:.4f}"
    )
    return metricas_avaliacao


def rodar_ensemble_bagging_disjunto(
    nome_arquitetura: str,
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float = TAXA_APRENDIZADO,
    num_epocas: int = NUM_EPOCAS,
    aplicar_aug: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    fracao_dados: float = 1.0,
    num_particoes: int = 2,
    seed_particao: int = SEED_SPLIT,
    estrategia_agregacao: str = ESTRATEGIA_AGREGACAO_PADRAO,
    reutilizar_checkpoints: bool = True,
    modelo_kwargs: Optional[Dict[str, Any]] = None,
    sufixo_nome: str = "",
    metadados_extras: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """
    Orquestra a Estratégia 2 (Ensemble Homogêneo com Bagging Disjunto):
    divide o conjunto de treino em `num_particoes` partições mutuamente
    exclusivas agrupadas por imagem-mãe, treina `num_particoes` instâncias da
    mesma arquitetura separadamente e avalia o ensemble agregado no teste.
    """
    if num_particoes < 2:
        raise ValueError("rodar_ensemble_bagging_disjunto requer num_particoes >= 2.")

    estrategia_norm = estrategia_agregacao.lower().strip()
    if estrategia_norm not in ESTRATEGIAS_AGREGACAO_ENSEMBLE:
        raise ValueError(
            f"Estratégia de agregação '{estrategia_agregacao}' inválida. "
            f"Opções válidas: {ESTRATEGIAS_AGREGACAO_ENSEMBLE}."
        )

    inicio_total = time.time()
    especificacoes_modelos: List[Tuple[str, str]] = []
    nomes_base_modelos: List[str] = []

    for idx_particao in range(num_particoes):
        sufixo_part = f"bag{idx_particao}de{num_particoes}_sp{seed_particao}"
        seed_instancia = seed + idx_particao

        nome_submodelo = _formatar_nome_experimento(
            nome_arquitetura=nome_arquitetura,
            nome_loss=nome_loss,
            nome_otimizador=nome_otimizador,
            nome_scheduler=nome_scheduler,
            nome_early_stopper=nome_early_stopper,
            lr=lr,
            aplicar_aug=aplicar_aug,
            seed=seed_instancia,
            sufixo_nome=sufixo_part,
        )
        caminho_ckpt = os.path.join(PATH_RESULTS_MODELOS, f"{nome_submodelo}.pt")

        if reutilizar_checkpoints and os.path.exists(caminho_ckpt):
            print(f"[CACHE] Reutilizando checkpoint existente da partição {idx_particao}: {caminho_ckpt}")
        else:
            metricas_sub = rodar_experimento_completo(
                nome_arquitetura=nome_arquitetura,
                nome_loss=nome_loss,
                nome_otimizador=nome_otimizador,
                nome_scheduler=nome_scheduler,
                nome_early_stopper=nome_early_stopper,
                lr=lr,
                num_epocas=num_epocas,
                aplicar_aug=aplicar_aug,
                num_workers=num_workers,
                seed=seed_instancia,
                fracao_dados=fracao_dados,
                num_particoes=num_particoes,
                indice_particao=idx_particao,
                seed_particao=seed_particao,
                sufixo_nome=sufixo_part,
                modelo_kwargs=modelo_kwargs,
                **kwargs,
            )
            if not metricas_sub or not os.path.exists(caminho_ckpt):
                raise RuntimeError(
                    f"Falha ao treinar a partição {idx_particao} ('{nome_submodelo}') no bagging disjunto."
                )

        if modelo_kwargs:
            especificacoes_modelos.append((nome_arquitetura, caminho_ckpt, modelo_kwargs))  # type: ignore[arg-type]
        else:
            especificacoes_modelos.append((nome_arquitetura, caminho_ckpt))
        nomes_base_modelos.append(nome_submodelo)

    _, _, dl_teste = criar_dataloaders(
        num_workers=num_workers,
        aplicar_aug=False,
        fracao_dados=fracao_dados,
    )

    metricas_avaliacao = avaliar_ensemble_checkpoints(
        especificacoes_modelos=especificacoes_modelos,
        dataloader_teste=dl_teste,
        estrategia_agregacao=estrategia_norm,
    )

    nome_ensemble = _formatar_nome_experimento(
        nome_arquitetura=f"ENS_BAG_{nome_arquitetura.upper()}_k{num_particoes}_{estrategia_norm}",
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        lr=lr,
        aplicar_aug=aplicar_aug,
        seed=seed,
        sufixo_nome=sufixo_nome,
    )

    metadados_finais: Dict[str, Any] = dict(metadados_extras) if metadados_extras else {}
    metadados_finais.update({
        "tipo_ensemble": "bagging_disjunto",
        "arquiteturas": nome_arquitetura.upper(),
        "num_particoes": num_particoes,
        "seed_particao": seed_particao,
        "estrategia_agregacao": estrategia_norm,
        "modelos_base": json.dumps(nomes_base_modelos),
    })

    tempo_total = time.time() - inicio_total
    salvar_resultado_completo(
        nome_modelo=nome_ensemble,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_epoca=num_epocas,
        melhor_mdice=metricas_avaliacao["mdice"],
        metricas_avaliacao=metricas_avaliacao,
        seed=seed,
        metadados_extras=metadados_finais,
    )

    salvar_tabela_experimentos(
        nome_modelo=nome_ensemble,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_mdice=metricas_avaliacao["mdice"],
        divergiu=False,
        tempo_treino=tempo_total,
    )

    print(
        f"\n[SUCESSO] Ensemble Bagging Disjunto '{nome_ensemble}' finalizado | "
        f"mDice: {metricas_avaliacao['mdice']:.4f} | mIoU: {metricas_avaliacao['miou']:.4f}"
    )
    return metricas_avaliacao


def rodar_ensemble_end_to_end(
    arquiteturas: Sequence[str],
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float = TAXA_APRENDIZADO,
    num_epocas: int = NUM_EPOCAS,
    aplicar_aug: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    fracao_dados: float = 1.0,
    pesos: Optional[Sequence[float]] = None,
    pesos_aprendiveis: Optional[bool] = None,
    modo_pesos: str = MODO_PESOS_ENSEMBLE_PADRAO,
    sufixo_nome: str = "",
    metadados_extras: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """
    Orquestra a Estratégia 3 (Ensemble End-to-End Ponderado):
    monta o nome padronizado `E2E_<ARQ1+ARQ2>_<fixo|aprendivel>_...`, treina
    conjuntamente via `rodar_experimento_completo` com `nome_arquitetura="ENSEMBLE_E2E"`
    e registra métricas e pesos normalizados finais nas tabelas CSV.
    """
    if not arquiteturas or len(arquiteturas) < 2:
        raise ValueError("rodar_ensemble_end_to_end requer pelo menos 2 arquiteturas.")

    if pesos_aprendiveis is None:
        modo_norm = modo_pesos.lower().strip()
        if modo_norm not in ("fixo", "aprendivel"):
            raise ValueError("modo_pesos deve ser 'fixo' ou 'aprendivel'.")
        usar_aprendivel = modo_norm == "aprendivel"
    else:
        usar_aprendivel = bool(pesos_aprendiveis)
        modo_norm = "aprendivel" if usar_aprendivel else "fixo"

    arqs_tag = "+".join(arq.upper() for arq in arquiteturas)
    nome_padronizado = _formatar_nome_experimento(
        nome_arquitetura=f"E2E_{arqs_tag}_{modo_norm}",
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        lr=lr,
        aplicar_aug=aplicar_aug,
        seed=seed,
        sufixo_nome=sufixo_nome,
    )

    modelo_kwargs: Dict[str, Any] = {
        "arquiteturas": list(arquiteturas),
        "pesos": list(pesos) if pesos is not None else None,
        "pesos_aprendiveis": usar_aprendivel,
    }
    if "submodelos_kwargs" in kwargs:
        modelo_kwargs["submodelos_kwargs"] = kwargs.pop("submodelos_kwargs")

    metadados_finais: Dict[str, Any] = dict(metadados_extras) if metadados_extras else {}
    metadados_finais.update({
        "tipo_ensemble": "end_to_end",
        "arquiteturas": arqs_tag,
        "modo_pesos": modo_norm,
    })

    return rodar_experimento_completo(
        nome_arquitetura="ENSEMBLE_E2E",
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        lr=lr,
        num_epocas=num_epocas,
        aplicar_aug=aplicar_aug,
        num_workers=num_workers,
        seed=seed,
        fracao_dados=fracao_dados,
        modelo_kwargs=modelo_kwargs,
        metadados_extras=metadados_finais,
        nome_modelo=nome_padronizado,
        **kwargs,
    )
