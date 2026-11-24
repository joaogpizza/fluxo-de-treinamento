"""Testes do pareamento determinístico e do split por imagem"""

import random

import pytest

from src.data.pareamento import (
    ParArquivos,
    dividir_pares,
    ordenar_deterministicamente,
    parear_arquivos
)


def _criar_arquivos(diretorio, nomes):
    diretorio.mkdir(parents=True, exist_ok=True)
    for nome in nomes:
        (diretorio / nome).write_bytes(b"x")


def _pares_falsos(quantidade):
    return [ParArquivos(f"img_{i:03d}", f"i{i}.tif", f"m{i}.png") for i in range(quantidade)]


def test_pareamento_ordena_e_ignora_ocultos(tmp_path):
    _criar_arquivos(tmp_path / "raw", ["b.tif", "a.TIF", "c.tif", ".DS_Store", "._a.tif", "notas.txt"])
    _criar_arquivos(tmp_path / "mascaras", ["c.png", "a.png", "b.png", ".DS_Store", "._a.png"])

    pares = parear_arquivos(str(tmp_path / "raw"), str(tmp_path / "mascaras"), ".tif", ".png")

    assert [par.nome_base for par in pares] == ["a", "b", "c"]
    assert pares[0].caminho_imagem.endswith("a.TIF")
    assert pares[0].caminho_mascara.endswith("a.png")


def test_pareamento_falha_com_imagem_sem_mascara(tmp_path):
    _criar_arquivos(tmp_path / "raw", ["a.tif", "b.tif"])
    _criar_arquivos(tmp_path / "mascaras", ["a.png"])

    with pytest.raises(ValueError, match="sem máscara"):
        parear_arquivos(str(tmp_path / "raw"), str(tmp_path / "mascaras"), ".tif", ".png")


def test_pareamento_falha_com_mascara_sem_imagem(tmp_path):
    _criar_arquivos(tmp_path / "raw", ["a.tif"])
    _criar_arquivos(tmp_path / "mascaras", ["a.png", "z.png"])

    with pytest.raises(ValueError, match="sem imagem"):
        parear_arquivos(str(tmp_path / "raw"), str(tmp_path / "mascaras"), ".tif", ".png")


def test_pareamento_falha_sem_nenhum_par(tmp_path):
    _criar_arquivos(tmp_path / "raw", [".DS_Store"])
    _criar_arquivos(tmp_path / "mascaras", [])

    with pytest.raises(FileNotFoundError):
        parear_arquivos(str(tmp_path / "raw"), str(tmp_path / "mascaras"), ".tif", ".png")


def test_divisao_e_deterministica_disjunta_e_completa():
    pares = _pares_falsos(100)

    divisao = dividir_pares(pares, 0.7, 0.15, 0.15, seed=42)
    random.seed(123)  # o estado global do random não pode influenciar o split
    divisao_repetida = dividir_pares(list(reversed(pares)), 0.7, 0.15, 0.15, seed=42)

    assert divisao == divisao_repetida
    assert [len(divisao[nome]) for nome in ("train", "val", "test")] == [70, 15, 15]

    nomes = [par.nome_base for nome in ("train", "val", "test") for par in divisao[nome]]
    assert len(nomes) == len(set(nomes)) == 100


def test_divisao_muda_com_a_seed():
    pares = _pares_falsos(100)

    divisao_a = dividir_pares(pares, 0.7, 0.15, 0.15, seed=1)
    divisao_b = dividir_pares(pares, 0.7, 0.15, 0.15, seed=2)

    assert divisao_a["test"] != divisao_b["test"]


def test_divisao_garante_val_e_teste_nao_vazios():
    divisao = dividir_pares(_pares_falsos(5), 0.9, 0.05, 0.05, seed=0)

    assert len(divisao["val"]) == 1 and len(divisao["test"]) == 1 and len(divisao["train"]) == 3


def test_divisao_rejeita_fracoes_invalidas_e_poucos_pares():
    with pytest.raises(ValueError):
        dividir_pares(_pares_falsos(10), 0.7, 0.2, 0.2, seed=0)
    with pytest.raises(ValueError):
        dividir_pares(_pares_falsos(2), 0.7, 0.15, 0.15, seed=0)


def test_ordenar_deterministicamente_nao_depende_da_ordem_de_entrada():
    nomes = [f"n{i}" for i in range(30)]

    ordem_a = ordenar_deterministicamente(nomes, chave=lambda nome: nome, seed=7)
    ordem_b = ordenar_deterministicamente(list(reversed(nomes)), chave=lambda nome: nome, seed=7)

    assert ordem_a == ordem_b
    assert sorted(ordem_a) == sorted(nomes)
