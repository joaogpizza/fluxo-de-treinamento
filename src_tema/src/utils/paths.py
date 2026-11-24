import os

# Definindo a raiz a partir do diretório atual do script (subindo um nível)
PATH_RAIZ = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
PATH_DESAFIO3 = os.path.join(
    os.path.dirname(PATH_RAIZ),
    "desafio3"
)

PATH_DATA = os.path.join(PATH_RAIZ, "data")
PATH_SCRIPTS = os.path.join(PATH_RAIZ, "scripts")
PATH_RESULTS = os.path.join(PATH_RAIZ, "results")

PATH_RAW = os.path.join(PATH_DATA, "raw")

# Dataset OEDB: raw/ (imagens .tif) e mascaras/ (máscaras .png), sem split prévio
PATH_OEDB = os.path.join(PATH_RAW, "OEDB")
PATH_OEDB_IMG = os.path.join(PATH_OEDB, "raw")
PATH_OEDB_MASK = os.path.join(PATH_OEDB, "mascaras")

# Lista de imagens (nome base) de cada split, gerada por scripts/criacao_patches.py
PATH_SPLIT_MANIFESTO = os.path.join(PATH_DATA, "split_oedb.csv")

PATH_PATCHES_IMGS = os.path.join(PATH_DATA, "patches_imgs")
PATH_PATCHES_IMGS_TRAIN = os.path.join(PATH_PATCHES_IMGS, "train")
PATH_PATCHES_IMGS_VAL = os.path.join(PATH_PATCHES_IMGS, "val")
PATH_PATCHES_IMGS_TEST = os.path.join(PATH_PATCHES_IMGS, "test")

PATH_PATCHES_GT = os.path.join(PATH_DATA, "patches_gt")
PATH_PATCHES_GT_TRAIN = os.path.join(PATH_PATCHES_GT, "train")
PATH_PATCHES_GT_VAL = os.path.join(PATH_PATCHES_GT, "val")
PATH_PATCHES_GT_TEST = os.path.join(PATH_PATCHES_GT, "test")

PATH_PLOTS = os.path.join(PATH_RESULTS, "plots")
PATH_PLOTS_CURVAS_APRENDIZADO = os.path.join(PATH_PLOTS, "curvas_aprendizado")
PATH_PLOTS_CURVAS_APRENDIZADO_LOSS = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "loss"
)
PATH_PLOTS_CURVAS_APRENDIZADO_MDICE = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "mdice"
)
PATH_PLOTS_CURVAS_APRENDIZADO_TAXA_APRENDIZADO = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "taxa_aprendizado"
)

PATH_RESULTS_TABELAS = os.path.join(
    PATH_RESULTS,
    "tabelas"
)
PATH_RESULTS_TABELAS_TABELA_COMPLETA = os.path.join(
    PATH_RESULTS_TABELAS,
    "tabela_completa.csv"
)

PATH_RESULTS_MODELOS = os.path.join(
    PATH_RESULTS,
    "modelos"
)

PATH_LOGS = os.path.join(PATH_DESAFIO3, "logs")
PATH_TABELA_EXPERIMENTOS = os.path.join(PATH_DESAFIO3, "experimentos.csv")
