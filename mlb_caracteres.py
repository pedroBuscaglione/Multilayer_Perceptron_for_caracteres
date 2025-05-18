# Pedro Serrano Buscaglione n° 14603652

import random
import math

# === Funções de utilidade para salvar saídas ===
def salvar_hiperparametros(input_size, hidden_size, output_size, lr, epochs, path="saidas/hiperparametros.txt"):
    with open(path, "w") as f:
        f.write(f"Tamanho da camada de entrada: {input_size}\n")
        f.write(f"Tamanho da camada oculta: {hidden_size}\n")
        f.write(f"Tamanho da camada de saída: {output_size}\n")
        f.write(f"Taxa de aprendizado: {lr}\n")
        f.write(f"Épocas: {epochs}\n")

def salvar_pesos(W1, b1, W2, b2, path):
    with open(f"saidas/{path}", "w") as f:
        f.write("Pesos W1:\n")
        for linha in W1:
            f.write(",".join(map(str, linha)) + "\n")
        f.write("Bias b1:\n")
        f.write(",".join(map(str, b1)) + "\n")
        f.write("Pesos W2:\n")
        for linha in W2:
            f.write(",".join(map(str, linha)) + "\n")
        f.write("Bias b2:\n")
        f.write(",".join(map(str, b2)) + "\n")

def salvar_erros_por_epoca(lista_erros, path):
    with open(f"saidas/{path}", "w") as f:
        for i, erro in enumerate(lista_erros):
            f.write(f"Epoch {i+1}: Erro = {erro:.4f}\n")

def salvar_saidas(X, Y_true, W1, b1, W2, b2, path):
    with open(f"saidas/{path}", "w") as f:
        for x, y_true in zip(X, Y_true):
            _, _, _, y_pred = forward_pass(x, W1, b1, W2, b2)
            pred_idx = y_pred.index(max(y_pred))
            letra_predita = chr(pred_idx + ord('A'))
            f.write(f"Entrada: {x}\n")
            f.write(f"Saída predita: {letra_predita} | Real: {y_true}\n\n")

# === Carregamento e processamento de dados ===
def load_data():
    with open("X.txt") as f:
        X = [list(map(int, line.strip().replace(',', ' ').split())) for line in f.readlines()]
    with open("Y_letra.txt") as f:
        Y = [line.strip().upper() for line in f.readlines()]
    return X, Y

def split_data(X, Y, train_ratio=0.8):
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)
    split_point = int(len(X) * train_ratio)
    return X[:split_point], Y[:split_point], X[split_point:], Y[split_point:]

def one_hot_encode_label(label):
    vec = [0] * 26
    index = ord(label.upper()) - ord('A')
    vec[index] = 1
    return vec

def relu(x): return [max(0, v) for v in x]

def relu_derivative(x): return [1 if v > 0 else 0 for v in x]

def softmax(z):
    max_z = max(z)
    exps = [math.exp(i - max_z) for i in z]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def initialize_weights(input_size, hidden_size, output_size):
    W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    W2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
    b2 = [0.0] * output_size
    return W1, b1, W2, b2

def forward_pass(x, W1, b1, W2, b2):
    z1 = [sum(w * xi for w, xi in zip(w_row, x)) + b for w_row, b in zip(W1, b1)]
    a1 = relu(z1)
    z2 = [sum(w * ai for w, ai in zip(w_row, a1)) + b for w_row, b in zip(W2, b2)]
    a2 = softmax(z2)
    return z1, a1, z2, a2

def train(X, Y, hidden_size, epochs, lr, W1, b1, W2, b2):
    erros = []
    input_size = len(X[0])
    output_size = 26
    for epoch in range(epochs):
        correct = 0
        for x, label in zip(X, Y):
            y_true = one_hot_encode_label(label)
            z1, a1, z2, y_pred = forward_pass(x, W1, b1, W2, b2)
            if y_pred.index(max(y_pred)) == y_true.index(1):
                correct += 1
            dL_dz2 = [y_pred[i] - y_true[i] for i in range(output_size)]
            dL_dW2 = [[dL_dz2[i] * a1[j] for j in range(hidden_size)] for i in range(output_size)]
            dL_db2 = dL_dz2[:]
            dL_da1 = [sum(dL_dz2[k] * W2[k][i] for k in range(output_size)) for i in range(hidden_size)]
            dL_dz1 = [dL_da1[i] * relu_derivative([z1[i]])[0] for i in range(hidden_size)]
            dL_dW1 = [[dL_dz1[i] * x[j] for j in range(input_size)] for i in range(hidden_size)]
            dL_db1 = dL_dz1[:]
            for i in range(output_size):
                for j in range(hidden_size):
                    W2[i][j] -= lr * dL_dW2[i][j]
                b2[i] -= lr * dL_db2[i]
            for i in range(hidden_size):
                for j in range(input_size):
                    W1[i][j] -= lr * dL_dW1[i][j]
                b1[i] -= lr * dL_db1[i]
        acc = correct / len(X)
        erros.append(1 - acc)
        print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")
    return W1, b1, W2, b2, erros

def evaluate(X, Y, W1, b1, W2, b2):
    y_true = []
    y_pred = []
    for x, label in zip(X, Y):
        _, _, _, a2 = forward_pass(x, W1, b1, W2, b2)
        pred_idx = a2.index(max(a2))
        y_true.append(label)
        y_pred.append(chr(pred_idx + ord('A')))
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    acc = correct / len(Y)
    print(f"Acurácia final: {acc:.4f}")
    print("\nMatriz de Confusão:")
    labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    matrix = [[0 for _ in labels] for _ in labels]
    label_index = {c: i for i, c in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        i = label_index[yt]
        j = label_index[yp]
        matrix[i][j] += 1
    print("      " + " ".join(labels))
    for i, row in enumerate(matrix):
        print(f"{labels[i]}: " + " ".join(f"{val:2d}" for val in row))
    return acc

def plot_erro_por_epoca(erros):
    with open("erro_por_epoca.txt", "w") as f:
        for i, erro in enumerate(erros):
            f.write(f"Epoch {i+1}: {erro:.4f}\n")
            
def k_fold_cross_validation(X, Y, k=5, patience=5, max_epochs=50, hidden_size=32, learning_rate=0.05):
    data = list(zip(X, Y))
    random.shuffle(data)
    X, Y = zip(*data)
    fold_size = len(X) // k
    total_accuracy = 0
    for fold in range(k):
        print(f"\n=== Fold {fold + 1}/{k} ===")
        start = fold * fold_size
        end = start + fold_size
        X_val = X[start:end]
        Y_val = Y[start:end]
        X_train = X[:start] + X[end:]
        Y_train = Y[:start] + Y[end:]
        W1, b1, W2, b2 = initialize_weights(len(X[0]), hidden_size, 26)
        salvar_pesos(W1, b1, W2, b2, f"pesos_iniciais_fold{fold+1}.txt")
        best_acc = 0
        no_improvement = 0
        best_weights = None
        erros_fold = []
        for epoch in range(max_epochs):
            W1, b1, W2, b2, erros = train(X_train, Y_train, hidden_size, 1, learning_rate, W1, b1, W2, b2)
            acc = evaluate(X_val, Y_val, W1, b1, W2, b2)
            erros_fold.extend(erros)
            print(f"Epoch {epoch + 1} | Val Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_weights = (W1, b1, W2, b2)
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("⏹️ Parada antecipada ativada.")
                    break
        W1, b1, W2, b2 = best_weights
        salvar_pesos(W1, b1, W2, b2, f"pesos_finais_fold{fold+1}.txt")
        salvar_erros_por_epoca(erros_fold, f"erros_por_epoca_fold{fold+1}.txt")
        salvar_saidas(X_val, Y_val, W1, b1, W2, b2, f"saidas_validacao_fold{fold+1}.txt")
        total_accuracy += best_acc
        print(f"Melhor acurácia neste fold: {best_acc:.4f}")
    media = total_accuracy / k
    with open("media_accuracies.txt", "w") as f:
        f.write(f"Média de acurácia (validação cruzada): {media:.4f}\n")

# === Execução principal ===
if __name__ == "__main__":
    X, Y = load_data()
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    input_size = len(X[0])
    hidden_size = 32
    output_size = 26
    lr = 0.05
    epochs = 50

    salvar_hiperparametros(input_size, hidden_size, output_size, lr, epochs)
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    salvar_pesos(W1, b1, W2, b2, "pesos_iniciais.txt")
    W1, b1, W2, b2, erros = train(X_train, Y_train, hidden_size, epochs, lr, W1, b1, W2, b2)
    salvar_pesos(W1, b1, W2, b2, "pesos_finais.txt")
    salvar_erros_por_epoca(erros, "erros_por_epoca.txt")
    salvar_saidas(X_test, Y_test, W1, b1, W2, b2, "saidas_teste.txt")
    evaluate(X_test, Y_test, W1, b1, W2, b2)

    print("\n=== Iniciando validação cruzada com parada antecipada ===")
    k_fold_cross_validation(X, Y, k=5, patience=5, max_epochs=50, hidden_size=32, learning_rate=0.05)
