"""
Lida com o pareamento determinístico entre imagens e máscaras e com a
divisão dos pares em treino, validação e teste
"""

import hashlib
import math
import os

from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Sequence, TypeVar

T = TypeVar("T")

NOMES_SPLITS = ("train", "val", "test")
MAX_NOMES_NA_MENSAGEM = 5


class ParArquivos(NamedTuple):
    """
    Par (imagem, máscara) identificado pelo nome base (Path.stem) do arquivo.
    """
    nome_base: str
    caminho_imagem: str
    caminho_mascara: str


def _resumir_nomes(nomes: Sequence[str]) -> str:
    """
    Formata os primeiros nomes de uma lista para uso em mensagens de erro.
    """
    if not nomes:
        return ""
    resumo = ", ".join(nomes[:MAX_NOMES_NA_MENSAGEM])
    if len(nomes) > MAX_NOMES_NA_MENSAGEM:
        resumo += ", ..."
    return f" [{resumo}]"


def _indexar_por_nome_base(diretorio: str, extensao: str) -> Dict[str, str]:
    """
    Mapeia nome base -> caminho dos arquivos de um diretório que têm a extensão
    informada (sem diferenciar maiúsculas de minúsculas).

    Arquivos ocultos (nomes iniciados por ".", como .DS_Store e os "._*" do
    macOS), subdiretórios e arquivos com outras extensões são ignorados.
    Levanta ValueError se dois arquivos tiverem o mesmo nome base.
    """
    if not os.path.isdir(diretorio):
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio}")

    extensao = extensao.lower()
    indice: Dict[str, str] = {}

    for arquivo in sorted(os.listdir(diretorio)):
        caminho = os.path.join(diretorio, arquivo)

        if arquivo.startswith(".") or not os.path.isfile(caminho):
            continue
        if Path(arquivo).suffix.lower() != extensao:
            continue

        nome_base = Path(arquivo).stem
        if nome_base in indice:
            raise ValueError(
                f"Nome base duplicado em {diretorio}: '{nome_base}' "
                f"({os.path.basename(indice[nome_base])} e {arquivo})."
            )
        indice[nome_base] = caminho

    return indice


def parear_arquivos(
    diretorio_imagens: str,
    diretorio_mascaras: str,
    extensao_imagem: str,
    extensao_mascara: str
) -> List[ParArquivos]:
    """
    Pareia <diretorio_imagens>/<nome><extensao_imagem> com
    <diretorio_mascaras>/<nome><extensao_mascara> pelo nome base.

    O retorno é ordenado pelo nome base. Levanta ValueError caso exista
    imagem sem máscara ou máscara sem imagem, e FileNotFoundError caso não
    exista nenhum par.
    """
    imagens = _indexar_por_nome_base(diretorio_imagens, extensao_imagem)
    mascaras = _indexar_por_nome_base(diretorio_mascaras, extensao_mascara)

    sem_mascara = sorted(set(imagens) - set(mascaras))
    sem_imagem = sorted(set(mascaras) - set(imagens))

    if sem_mascara or sem_imagem:
        raise ValueError(
            "Pareamento inconsistente entre imagens e máscaras: "
            f"{len(sem_mascara)} imagem(ns) sem máscara{_resumir_nomes(sem_mascara)} e "
            f"{len(sem_imagem)} máscara(s) sem imagem{_resumir_nomes(sem_imagem)}."
        )

    if not imagens:
        raise FileNotFoundError(
            f"Nenhum par imagem/máscara encontrado em '{diretorio_imagens}' "
            f"({extensao_imagem}) e '{diretorio_mascaras}' ({extensao_mascara})."
        )

    return [
        ParArquivos(nome, imagens[nome], mascaras[nome])
        for nome in sorted(imagens)
    ]


def ordenar_deterministicamente(
    itens: Sequence[T],
    chave: Callable[[T], str],
    seed: int
) -> List[T]:
    """
    Embaralha os itens de forma reproduzível.

    A ordem depende apenas de (seed, chave): não usa o estado global dos
    geradores aleatórios e é igual entre plataformas e versões de
    Python/NumPy/PyTorch.
    """
    def impressao_digital(item: T) -> tuple:
        nome = chave(item)
        digest = hashlib.sha256(f"{seed}:{nome}".encode("utf-8")).hexdigest()
        return digest, nome

    return sorted(itens, key=impressao_digital)


def _validar_sem_vazamento(
    divisao: Dict[str, List[ParArquivos]],
    total: int
) -> None:
    """
    Garante que nenhuma imagem aparece em mais de um split e que nenhuma foi perdida.
    """
    vistos: Dict[str, str] = {}
    for nome_split, pares in divisao.items():
        for par in pares:
            if par.nome_base in vistos:
                raise ValueError(
                    f"Vazamento de dados: '{par.nome_base}' está em "
                    f"'{vistos[par.nome_base]}' e em '{nome_split}'."
                )
            vistos[par.nome_base] = nome_split

    if len(vistos) != total:
        raise ValueError("A divisão perdeu imagens: o total dos splits difere do total de pares.")


def dividir_pares(
    pares: Sequence[ParArquivos],
    fracao_treino: float,
    fracao_val: float,
    fracao_teste: float,
    seed: int
) -> Dict[str, List[ParArquivos]]:
    """
    Divide os pares em treino ("train"), validação ("val") e teste ("test").

    A divisão é feita por imagem (um par nunca aparece em dois splits) e é
    determinística: depende só dos nomes base, das frações e de seed.
    Validação e teste recebem pelo menos um par cada. Cada split é retornado
    ordenado pelo nome base.
    """
    fracoes = (fracao_treino, fracao_val, fracao_teste)
    if any(not 0.0 < fracao < 1.0 for fracao in fracoes) or not math.isclose(
        sum(fracoes), 1.0, abs_tol=1e-6
    ):
        raise ValueError("As frações de treino, validação e teste devem estar em (0, 1) e somar 1.")

    total = len(pares)
    if len({par.nome_base for par in pares}) != total:
        raise ValueError("Existem pares com nome base repetido.")
    if total < 3:
        raise ValueError(
            f"São necessários pelo menos 3 pares para dividir em treino, validação e teste (recebidos {total})."
        )

    num_val = max(1, round(total * fracao_val))
    num_teste = max(1, round(total * fracao_teste))
    num_treino = total - num_val - num_teste
    if num_treino < 1:
        raise ValueError("Poucos pares para as frações escolhidas: o split de treino ficaria vazio.")

    embaralhados = ordenar_deterministicamente(pares, lambda par: par.nome_base, seed)

    divisao = {
        "train": embaralhados[:num_treino],
        "val": embaralhados[num_treino:num_treino + num_val],
        "test": embaralhados[num_treino + num_val:],
    }
    _validar_sem_vazamento(divisao, total)

    return {
        nome_split: sorted(lista, key=lambda par: par.nome_base)
        for nome_split, lista in divisao.items()
    }
