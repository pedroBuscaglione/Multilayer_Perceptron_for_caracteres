import matplotlib.pyplot as plt
import os

def plot_acuracias_folds(k=5, pasta=".", arquivo_media="saidas/media_accuracies.txt"):
    acuracias = []

    for i in range(1, k+1):
        caminho = os.path.join(pasta, f"erros_por_epoca_fold{i}.txt")
        with open(caminho) as f:
            ultimas_linhas = f.readlines()
            # menor erro = maior acurácia
            erros = [float(linha.strip().split('=')[1]) for linha in ultimas_linhas]
            acuracia = 1 - min(erros)
            acuracias.append(acuracia)

    plt.bar(range(1, k+1), acuracias, color="skyblue")
    plt.xlabel("Fold")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por Fold (Validação Cruzada)")
    plt.ylim(0, 1)
    plt.xticks(range(1, k+1))
    plt.tight_layout()
    plt.savefig("grafico_acuracia_por_fold.png")
    plt.show()

if __name__ == "__main__":
    plot_acuracias_folds()
