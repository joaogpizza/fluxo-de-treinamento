"""Testes da leitura (.tif/.png) e do recorte em patches"""

import numpy as np

from PIL import Image

from configs.basicas import TAM_PATCH
from scripts.criacao_patches import (
    binarizar_mascara,
    gerar_patches,
    ler_imagem_bgr,
    ler_mascara_png
)


def test_tif_rgba_vira_bgr_de_3_canais(tmp_path):
    rgba = np.zeros((40, 50, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 200, 100, 50, 7
    caminho = tmp_path / "a.tif"
    Image.fromarray(rgba, mode="RGBA").save(caminho)

    imagem = ler_imagem_bgr(str(caminho))

    assert imagem.shape == (40, 50, 3) and imagem.dtype == np.uint8
    assert tuple(imagem[0, 0]) == (50, 100, 200)  # BGR, cores intactas mesmo com alfa baixo


def test_tif_em_tons_de_cinza_e_replicado_em_3_canais(tmp_path):
    cinza = np.full((32, 32), 90, dtype=np.uint8)
    caminho = tmp_path / "g.tif"
    Image.fromarray(cinza, mode="L").save(caminho)

    imagem = ler_imagem_bgr(str(caminho))

    assert imagem.shape == (32, 32, 3) and set(np.unique(imagem)) == {90}


def test_mascara_aceita_0_1_0_255_e_paleta(tmp_path):
    base = np.zeros((8, 8), dtype=np.uint8)
    base[2:4, 2:4] = 1

    Image.fromarray(base, mode="L").save(tmp_path / "bin01.png")
    Image.fromarray(base * 255, mode="L").save(tmp_path / "bin255.png")
    paleta = Image.fromarray(base, mode="P")
    paleta.putpalette([0, 0, 0, 255, 255, 255] + [0] * 756)
    paleta.save(tmp_path / "paleta.png")

    for nome in ("bin01", "bin255", "paleta"):
        mascara = binarizar_mascara(ler_mascara_png(str(tmp_path / f"{nome}.png")))
        assert set(np.unique(mascara)) == {0, 255}
        assert int((mascara > 0).sum()) == 4


def test_patches_sem_sobra_cobrem_a_imagem_inteira():
    imagem = np.random.randint(0, 255, (2 * TAM_PATCH, 3 * TAM_PATCH, 3), dtype=np.uint8)
    mascara = np.random.randint(0, 2, (2 * TAM_PATCH, 3 * TAM_PATCH), dtype=np.uint8) * 255

    patches = list(gerar_patches(imagem, mascara))

    assert len(patches) == 6
    assert all(img.shape == (TAM_PATCH, TAM_PATCH, 3) and msk.shape == (TAM_PATCH, TAM_PATCH) for img, msk in patches)
    assert np.array_equal(patches[0][0], imagem[:TAM_PATCH, :TAM_PATCH])


def test_sobra_grande_na_borda_vira_janela_encostada_sem_padding():
    # largura = 1.75 patches: janelas em x = 0 e x = largura - TAM_PATCH (sobreposição), altura = 1 patch
    largura = TAM_PATCH + (3 * TAM_PATCH) // 4
    imagem = np.random.randint(1, 255, (TAM_PATCH, largura, 3), dtype=np.uint8)  # sem pixels pretos
    mascara = np.full((TAM_PATCH, largura), 255, dtype=np.uint8)

    patches = list(gerar_patches(imagem, mascara))

    assert len(patches) == 2
    assert np.array_equal(patches[0][0], imagem[:, :TAM_PATCH])
    assert np.array_equal(patches[1][0], imagem[:, largura - TAM_PATCH:])
    assert all(img.min() >= 1 and msk.min() == 255 for img, msk in patches)  # nenhum zero de padding


def test_sobra_pequena_na_borda_e_descartada():
    imagem = np.full((TAM_PATCH + 10, TAM_PATCH, 3), 7, dtype=np.uint8)
    mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)

    patches = list(gerar_patches(imagem, mascara))

    assert len(patches) == 1
    assert np.array_equal(patches[0][0], imagem[:TAM_PATCH])


def test_sobra_exatamente_metade_do_patch_gera_janela_extra():
    lado = TAM_PATCH + TAM_PATCH // 2
    imagem = np.full((lado, lado, 3), 255, dtype=np.uint8)
    mascara = np.full((lado, lado), 255, dtype=np.uint8)

    patches = list(gerar_patches(imagem, mascara))

    assert len(patches) == 4
    assert all(img.min() == 255 for img, _ in patches)


def test_imagem_menor_que_o_patch_e_completada_com_zeros_se_cobrir_metade():
    imagem = np.full((TAM_PATCH - 10, TAM_PATCH - 10, 3), 10, dtype=np.uint8)
    mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)

    patches = list(gerar_patches(imagem, mascara))

    assert len(patches) == 1
    assert patches[0][0][TAM_PATCH - 10:].max() == 0 and patches[0][0][:TAM_PATCH - 10, :TAM_PATCH - 10].min() == 10


def test_imagem_muito_menor_que_o_patch_nao_gera_patch():
    imagem = np.full((TAM_PATCH // 4, TAM_PATCH // 4, 3), 10, dtype=np.uint8)
    mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)

    assert list(gerar_patches(imagem, mascara)) == []
