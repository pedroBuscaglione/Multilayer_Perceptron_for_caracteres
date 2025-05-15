import matplotlib.pyplot as plt

def plot_erro_por_epoca(caminho="saidas/erros_por_epoca.txt"):
    epocas = []
    erros = []

    with open(caminho) as f:
        for linha in f:
            if "Epoch" in linha:
                partes = linha.strip().split(":")
                epoca = int(partes[0].split()[1])
                erro = float(partes[1].split("=")[1])
                epocas.append(epoca)
                erros.append(erro)

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, erros, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Erro (1 - acurácia)")
    plt.title("Erro por época (treinamento simples)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_erro_por_epoca.png")
    plt.show()

if __name__ == "__main__":
    plot_erro_por_epoca()
