"""
Gera os patches de treino, validação e teste do dataset OEDB.

Fluxo:
1. Pareia OEDB/raw/<nome>.tif com OEDB/mascaras/<nome>.png pelo nome base
   (falha imediatamente se houver imagem sem máscara ou vice-versa).
2. Divide as IMAGENS (e não os patches) em treino, validação e teste de forma
   determinística (SEED_SPLIT). Assim, patches de uma mesma imagem nunca caem
   em splits diferentes, o que causaria vazamento de dados.
3. Recorta cada imagem em patches de TAM_PATCH x TAM_PATCH, converte a imagem
   para RGB de 8 bits, binariza a máscara em {0, 255} e salva tudo em PNG.
4. Imprime um relatório (proporção de foreground, valores encontrados nas
   máscaras e média/desvio das cores no treino) para conferir se as
   configurações em configs/basicas.py ainda fazem sentido.

Uso (da raiz src_tema, como antes):
    python -m scripts.criacao_patches [--limpar]

Se os diretórios de patches já tiverem arquivos (de outro dataset ou de outro
split), o script falha; use --limpar para apagá-los antes de gerar os novos.
"""

import argparse
import csv
import os
import sys

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from configs.basicas import (
    TAM_PATCH,
    STRIDE_PATCH,
    MIN_FRACAO_VALIDA_PATCH,
    EXTENSAO_IMAGEM_RAW,
    EXTENSAO_MASCARA_RAW,
    EXTENSAO_PATCH_IMAGEM,
    EXTENSAO_PATCH_MASCARA,
    NORMALIZACAO_MEAN,
    NORMALIZACAO_STD,
    SEED_SPLIT,
    FRACAO_TREINO,
    FRACAO_VAL,
    FRACAO_TESTE
)
from src.data.pareamento import NOMES_SPLITS, ParArquivos, parear_arquivos, dividir_pares
from src.utils.paths import (
    PATH_RAIZ,
    PATH_OEDB_IMG,
    PATH_OEDB_MASK,
    PATH_SPLIT_MANIFESTO,
    PATH_PATCHES_IMGS_TRAIN,
    PATH_PATCHES_IMGS_VAL,
    PATH_PATCHES_IMGS_TEST,
    PATH_PATCHES_GT_TRAIN,
    PATH_PATCHES_GT_VAL,
    PATH_PATCHES_GT_TEST
)

# Diferença máxima aceita entre a média/desvio do OEDB (treino) e os valores
# configurados antes de sugerir a atualização da normalização
TOLERANCIA_NORMALIZACAO = 0.05
# Linhas processadas por vez no cálculo das estatísticas de cor (limita a memória)
LINHAS_POR_LOTE_ESTATISTICAS = 512
# Modos do Pillow com mais de 8 bits por canal (a conversão para RGB os saturaria)
MODOS_PILLOW_NAO_8_BITS = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")

# Imagens de satélite podem passar do limite padrão do Pillow (proteção contra
# "decompression bomb"); o dataset é local e confiável.
Image.MAX_IMAGE_PIXELS = None

DIRETORIOS_SAIDA: Dict[str, Tuple[str, str]] = {
    "train": (PATH_PATCHES_IMGS_TRAIN, PATH_PATCHES_GT_TRAIN),
    "val": (PATH_PATCHES_IMGS_VAL, PATH_PATCHES_GT_VAL),
    "test": (PATH_PATCHES_IMGS_TEST, PATH_PATCHES_GT_TEST),
}


@dataclass
class EstatisticasSplit:
    """Acumula contagens e estatísticas usadas para validar o dataset."""
    num_imagens: int = 0
    num_patches: int = 0
    patches_sem_foreground: int = 0
    pixels_imagem: int = 0
    soma_canais_bgr: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    soma_quad_canais_bgr: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    histograma_mascara: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))

    def registrar_imagem(
        self,
        imagem_bgr: np.ndarray,
        mascara_bruta: np.ndarray,
        coletar_cor: bool
    ) -> None:
        """Registra o histograma da máscara e, opcionalmente, as estatísticas de cor."""
        self.num_imagens += 1
        self.histograma_mascara += np.bincount(mascara_bruta.ravel(), minlength=256)

        if not coletar_cor:
            return

        for inicio in range(0, imagem_bgr.shape[0], LINHAS_POR_LOTE_ESTATISTICAS):
            lote = imagem_bgr[inicio:inicio + LINHAS_POR_LOTE_ESTATISTICAS]
            pixels = lote.reshape(-1, 3).astype(np.float64)
            self.soma_canais_bgr += pixels.sum(axis=0)
            self.soma_quad_canais_bgr += np.square(pixels).sum(axis=0)
        self.pixels_imagem += imagem_bgr.shape[0] * imagem_bgr.shape[1]

    def registrar_patch(self, patch_mascara: np.ndarray) -> None:
        """Conta o patch e se ele possui algum pixel de foreground."""
        self.num_patches += 1
        if not patch_mascara.any():
            self.patches_sem_foreground += 1

    @property
    def fracao_foreground(self) -> float:
        """Fração de pixels de foreground nas máscaras originais."""
        total = int(self.histograma_mascara.sum())
        if total == 0:
            return 0.0
        return 1.0 - float(self.histograma_mascara[0]) / total

    @property
    def fracao_patches_sem_foreground(self) -> float:
        """Fração de patches cuja máscara é só fundo."""
        if self.num_patches == 0:
            return 0.0
        return self.patches_sem_foreground / self.num_patches

    def media_std_rgb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Média e desvio padrão por canal (ordem RGB, escala [0, 1])."""
        media_bgr = self.soma_canais_bgr / self.pixels_imagem
        variancia_bgr = self.soma_quad_canais_bgr / self.pixels_imagem - np.square(media_bgr)
        std_bgr = np.sqrt(np.maximum(variancia_bgr, 0.0))
        return media_bgr[::-1] / 255.0, std_bgr[::-1] / 255.0


def preparar_diretorios_saida(limpar: bool) -> None:
    """
    Garante que todas as pastas de saída existam e estejam vazias.

    Patches antigos (de outro dataset ou de outro split) misturados aos novos
    invalidariam a separação entre treino, validação e teste.
    """
    ocupados: Dict[str, List[str]] = {}

    for diretorio_imgs, diretorio_masks in DIRETORIOS_SAIDA.values():
        for diretorio in (diretorio_imgs, diretorio_masks):
            os.makedirs(diretorio, exist_ok=True)
            arquivos = [
                arquivo for arquivo in os.listdir(diretorio)
                if not arquivo.startswith(".") and os.path.isfile(os.path.join(diretorio, arquivo))
            ]
            if arquivos:
                ocupados[diretorio] = arquivos

    if not ocupados:
        return

    if not limpar:
        detalhes = "\n".join(f"  - {diretorio} ({len(arquivos)} arquivos)" for diretorio, arquivos in ocupados.items())
        raise FileExistsError(
            "Os diretórios de patches já possuem arquivos:\n"
            f"{detalhes}\n"
            "Execute novamente com --limpar para apagá-los antes de gerar os patches do OEDB."
        )

    for diretorio, arquivos in ocupados.items():
        for arquivo in arquivos:
            os.remove(os.path.join(diretorio, arquivo))


def salvar_manifesto_split(divisao: Dict[str, List[ParArquivos]]) -> None:
    """Salva em CSV a qual split cada imagem pertence, para auditoria e reprodutibilidade."""
    os.makedirs(os.path.dirname(PATH_SPLIT_MANIFESTO), exist_ok=True)

    with open(PATH_SPLIT_MANIFESTO, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["split", "nome_base", "seed_split"])
        for nome_split in NOMES_SPLITS:
            for par in divisao[nome_split]:
                escritor.writerow([nome_split, par.nome_base, SEED_SPLIT])


def _ler_imagem_com_pillow(caminho: str) -> Optional[np.ndarray]:
    """
    Tenta ler a imagem com o Pillow, retornando uint8 (H, W, 3) em ordem BGR.

    Retorna None se o Pillow não conseguir abrir o arquivo ou se ele tiver mais
    de 8 bits por canal.
    """
    try:
        with Image.open(caminho) as imagem_pil:
            if imagem_pil.mode in MODOS_PILLOW_NAO_8_BITS:
                return None
            imagem_rgb = np.array(imagem_pil.convert("RGB"))
    except Exception:
        return None

    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)


def ler_imagem_bgr(caminho: str) -> np.ndarray:
    """
    Lê uma imagem (ex: .tif) como array uint8 (H, W, 3) em ordem BGR.

    A conversão para 3 canais é explícita: canal alfa e canais extras são
    descartados e imagens em tons de cinza são replicadas nos 3 canais.

    O Pillow é usado primeiro porque descarta o alfa sem alterar as cores (o
    OpenCV multiplica as cores pelo alfa ao ler TIFF RGBA, escurecendo a
    imagem). O OpenCV fica como alternativa para TIFFs que o Pillow não abre
    ou que têm 16 bits por canal (reduzidos para 8 bits). Nesse caso, confira
    visualmente alguns patches gerados. A leitura via bytes funciona com
    caminhos com acentos no Windows.
    """
    imagem = _ler_imagem_com_pillow(caminho)

    if imagem is None:
        bytes_arquivo = np.fromfile(caminho, dtype=np.uint8)
        imagem = cv2.imdecode(bytes_arquivo, cv2.IMREAD_COLOR) if bytes_arquivo.size else None

    if imagem is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {caminho}")

    if imagem.ndim != 3 or imagem.shape[2] != 3 or imagem.dtype != np.uint8:
        raise ValueError(
            f"Imagem fora do formato esperado (H, W, 3) uint8: {caminho} "
            f"(shape={imagem.shape}, dtype={imagem.dtype})."
        )
    return imagem


def ler_mascara_png(caminho: str) -> np.ndarray:
    """
    Lê uma máscara .png como array uint8 (H, W) com os valores originais.

    Máscaras com paleta (modo "P") usam os índices da paleta; as demais são
    convertidas para tons de cinza de 8 bits (canal alfa ignorado).
    """
    try:
        with Image.open(caminho) as mascara_pil:
            if mascara_pil.mode == "P":
                valores = np.array(mascara_pil)
            else:
                valores = np.array(mascara_pil.convert("L"))
    except Exception as erro:
        raise RuntimeError(f"Não foi possível ler a máscara: {caminho}") from erro

    return valores.astype(np.uint8, copy=False)


def binarizar_mascara(mascara_bruta: np.ndarray) -> np.ndarray:
    """
    Converte a máscara para {0, 255}: qualquer valor maior que 0 é foreground.

    É o mesmo critério que HistologiaDataset aplica ao ler os patches
    ((mascara > 0) -> {0., 1.}).
    """
    return (mascara_bruta > 0).astype(np.uint8) * 255


def gerar_patches(
    imagem: np.ndarray,
    mascara: np.ndarray
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Recorta imagem e máscara em janelas TAM_PATCH x TAM_PATCH com passo STRIDE_PATCH.

    Janelas na borda que ultrapassam a imagem são completadas com zeros
    (imagem preta e máscara de fundo), mas só viram patch se pelo menos
    MIN_FRACAO_VALIDA_PATCH da área da janela for imagem real.
    """
    altura, largura = imagem.shape[:2]

    for y in range(0, altura, STRIDE_PATCH):
        for x in range(0, largura, STRIDE_PATCH):
            altura_valida = min(TAM_PATCH, altura - y)
            largura_valida = min(TAM_PATCH, largura - x)

            fracao_valida = (altura_valida * largura_valida) / (TAM_PATCH * TAM_PATCH)
            if fracao_valida < MIN_FRACAO_VALIDA_PATCH:
                continue

            if altura_valida == TAM_PATCH and largura_valida == TAM_PATCH:
                yield imagem[y:y + TAM_PATCH, x:x + TAM_PATCH], mascara[y:y + TAM_PATCH, x:x + TAM_PATCH]
                continue

            patch_imagem = np.zeros((TAM_PATCH, TAM_PATCH, 3), dtype=np.uint8)
            patch_mascara = np.zeros((TAM_PATCH, TAM_PATCH), dtype=np.uint8)
            patch_imagem[:altura_valida, :largura_valida] = imagem[y:y + altura_valida, x:x + largura_valida]
            patch_mascara[:altura_valida, :largura_valida] = mascara[y:y + altura_valida, x:x + largura_valida]
            yield patch_imagem, patch_mascara


def salvar_png(caminho: str, array: np.ndarray) -> None:
    """Salva um array como PNG (compatível com caminhos com acentos no Windows)."""
    sucesso, buffer = cv2.imencode(".png", array)
    if not sucesso:
        raise OSError(f"Falha ao codificar o PNG: {caminho}")
    buffer.tofile(caminho)


def processar_split(
    nome_split: str,
    pares: List[ParArquivos],
    dir_out_img: str,
    dir_out_mask: str
) -> EstatisticasSplit:
    """
    Gera os patches de todas as imagens de um split e retorna as estatísticas.
    """
    estatisticas = EstatisticasSplit()

    print(f"\n[{nome_split.upper()}] Processando {len(pares)} imagens...")

    for par in tqdm(
        pares,
        desc=f"Gerando patches - {nome_split}",
        unit="img",
        ncols=100,
        ascii=False,
        file=sys.stdout
    ):
        imagem = ler_imagem_bgr(par.caminho_imagem)
        mascara_bruta = ler_mascara_png(par.caminho_mascara)

        if imagem.shape[:2] != mascara_bruta.shape:
            raise ValueError(
                f"Imagem e máscara de '{par.nome_base}' têm dimensões diferentes: "
                f"{imagem.shape[:2]} e {mascara_bruta.shape}."
            )

        estatisticas.registrar_imagem(imagem, mascara_bruta, coletar_cor=(nome_split == "train"))
        mascara = binarizar_mascara(mascara_bruta)

        patches_gerados = 0
        for patch_imagem, patch_mascara in gerar_patches(imagem, mascara):
            nome_patch = f"{par.nome_base}_p{patches_gerados:04d}"
            salvar_png(os.path.join(dir_out_img, f"{nome_patch}{EXTENSAO_PATCH_IMAGEM}"), patch_imagem)
            salvar_png(os.path.join(dir_out_mask, f"{nome_patch}{EXTENSAO_PATCH_MASCARA}"), patch_mascara)
            estatisticas.registrar_patch(patch_mascara)
            patches_gerados += 1

        if patches_gerados == 0:
            tqdm.write(
                f"Aviso: Imagem {par.nome_base} {imagem.shape[:2]} não gerou patches "
                f"(menor que o patch {TAM_PATCH}x{TAM_PATCH})."
            )

    if estatisticas.num_patches == 0:
        raise RuntimeError(f"O split '{nome_split}' não gerou nenhum patch.")

    return estatisticas


def imprimir_relatorio(estatisticas: Dict[str, EstatisticasSplit]) -> None:
    """
    Resume os splits e sinaliza se as máscaras ou a normalização parecem inadequadas.
    """
    print("\n" + "=" * 70)
    print("RELATÓRIO DO DATASET OEDB")
    print("=" * 70)

    for nome_split in NOMES_SPLITS:
        stats = estatisticas[nome_split]
        print(
            f"[{nome_split.upper():<5}] imagens: {stats.num_imagens:>5} | patches: {stats.num_patches:>7} | "
            f"patches só com fundo: {stats.fracao_patches_sem_foreground:6.1%} | "
            f"pixels de foreground: {stats.fracao_foreground:6.2%}"
        )

    histograma_total = sum(stats.histograma_mascara for stats in estatisticas.values())
    valores_presentes = np.flatnonzero(histograma_total)
    total_pixels = int(histograma_total.sum())
    resumo_valores = ", ".join(
        f"{int(valor)} ({histograma_total[valor] / total_pixels:.2%})" for valor in valores_presentes[:10]
    )
    print(f"\nValores de pixel encontrados nas máscaras originais: {resumo_valores}")
    if len(valores_presentes) > 2:
        print(
            "ATENÇÃO: as máscaras têm mais de 2 valores distintos. Se forem codificações diferentes do mesmo "
            "foreground (ex: 1 e 255), a binarização '> 0' já resolve. Se forem classes distintas, o contrato "
            "binário do projeto (1 canal, BCE/Dice, métricas com classes 0 e 1) NÃO se aplica: o número de "
            "classes, a loss e as métricas precisariam ser ajustados."
        )

    media, std = estatisticas["train"].media_std_rgb()
    media_config = np.array(NORMALIZACAO_MEAN)
    std_config = np.array(NORMALIZACAO_STD)
    print("\nNormalização (treino, ordem RGB, escala [0, 1]):")
    print(f"  OEDB       mean={tuple(round(float(v), 4) for v in media)} std={tuple(round(float(v), 4) for v in std)}")
    print(f"  Configurado mean={NORMALIZACAO_MEAN} std={NORMALIZACAO_STD}")

    diferenca = max(float(np.abs(media - media_config).max()), float(np.abs(std - std_config).max()))
    if diferenca > TOLERANCIA_NORMALIZACAO:
        print(
            "ATENÇÃO: a estatística do OEDB difere dos valores configurados (ImageNet) em mais de "
            f"{TOLERANCIA_NORMALIZACAO}. Considere atualizar NORMALIZACAO_MEAN e NORMALIZACAO_STD em "
            "configs/basicas.py (isto altera os resultados e deve ser decidido conscientemente)."
        )


def main(limpar: bool = False) -> None:
    print("Iniciando rotina de fatiamento (Patch Cropping) - Dataset OEDB")
    print(f"Diretório Raiz Identificado: {PATH_RAIZ}")
    print(f"Tamanho do Patch: {TAM_PATCH}x{TAM_PATCH} | Stride: {STRIDE_PATCH}")
    print(
        f"Split por imagem (treino/val/teste): {FRACAO_TREINO:.0%}/{FRACAO_VAL:.0%}/{FRACAO_TESTE:.0%} "
        f"| Seed do split: {SEED_SPLIT}\n"
    )

    pares = parear_arquivos(PATH_OEDB_IMG, PATH_OEDB_MASK, EXTENSAO_IMAGEM_RAW, EXTENSAO_MASCARA_RAW)
    print(f"{len(pares)} pares imagem/máscara encontrados e pareados.")

    divisao = dividir_pares(pares, FRACAO_TREINO, FRACAO_VAL, FRACAO_TESTE, SEED_SPLIT)

    preparar_diretorios_saida(limpar)
    salvar_manifesto_split(divisao)

    estatisticas: Dict[str, EstatisticasSplit] = {}
    for nome_split in NOMES_SPLITS:
        dir_out_img, dir_out_mask = DIRETORIOS_SAIDA[nome_split]
        estatisticas[nome_split] = processar_split(
            nome_split=nome_split,
            pares=divisao[nome_split],
            dir_out_img=dir_out_img,
            dir_out_mask=dir_out_mask
        )

    imprimir_relatorio(estatisticas)
    print("\nProcessamento finalizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera os patches de treino, validação e teste do OEDB.")
    parser.add_argument(
        "--limpar",
        action="store_true",
        help="Apaga os arquivos já existentes nos diretórios de patches antes de gerar os novos."
    )
    argumentos = parser.parse_args()
    main(limpar=argumentos.limpar)
