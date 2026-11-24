""" Definição de algumas configurações básicas """

import torch.cuda as cuda

TAM_PATCH = 256
NORMALIZACAO_MEAN = (0.485, 0.456, 0.406)
NORMALIZACAO_STD = (0.229, 0.224, 0.225)
NUM_EPOCAS = 25
LIMIAR = 0.5
DEVICE = "cuda" if cuda.is_available() else "cpu"
TAXA_APRENDIZADO = 1e-3
TAM_BATCH = 16

# Dataset OEDB: extensões dos arquivos originais (raw/ e mascaras/) e dos
# patches gerados por scripts/criacao_patches.py
EXTENSAO_IMAGEM_RAW = ".tif"
EXTENSAO_MASCARA_RAW = ".png"
EXTENSAO_PATCH_IMAGEM = ".png"
EXTENSAO_PATCH_MASCARA = ".png"

# Geração dos patches. Uma janela só vira patch se pelo menos
# MIN_FRACAO_VALIDA_PATCH da sua área cobrir pixels reais da imagem; o
# restante é preenchido com zeros (imagem preta e máscara de fundo).
STRIDE_PATCH = TAM_PATCH
MIN_FRACAO_VALIDA_PATCH = 0.5

# Split treino/validação/teste do OEDB. A divisão é feita por IMAGEM, antes
# de gerar os patches, e depende apenas de SEED_SPLIT (e não da seed do
# experimento), para que todos os experimentos usem exatamente os mesmos splits.
SEED_SPLIT = 42
FRACAO_TREINO = 0.55
FRACAO_VAL = 0.15
FRACAO_TESTE = 0.3
