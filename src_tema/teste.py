import os
import torch
from collections import Counter
from PIL import Image

from src.data.dataloader import criar_dataloaders
from src.utils.paths import PATH_OEDB_IMG

# 1) estatística do treino inteiro, como o modelo enxerga (sem augmentation)
dl_treino, _, _ = criar_dataloaders(num_workers=0, aplicar_aug=False, fracao_dados=1.0)
soma = torch.zeros(3, dtype=torch.float64)
soma_quad = torch.zeros(3, dtype=torch.float64)
n = 0
for imagens, _ in dl_treino:
    x = imagens.double()
    soma += x.sum(dim=(0, 2, 3))
    soma_quad += (x ** 2).sum(dim=(0, 2, 3))
    n += x.shape[0] * x.shape[2] * x.shape[3]
media = soma / n
std = (soma_quad / n - media ** 2).sqrt()
print("mean por canal:", media.tolist(), "| std por canal:", std.tolist())

# 2) dimensões (largura, altura) dos .tif originais
tamanhos = Counter(
    Image.open(os.path.join(PATH_OEDB_IMG, f)).size
    for f in os.listdir(PATH_OEDB_IMG)
    if f.lower().endswith(".tif") and not f.startswith(".")
)
print(tamanhos.most_common(5))
