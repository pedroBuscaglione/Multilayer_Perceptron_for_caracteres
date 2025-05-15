# === Carregamento dos dados ===
def load_data():
    with open("X.txt") as f:
        X = [list(map(int, line.strip().replace(',', ' ').split())) for line in f.readlines()]
    with open("Y_letra.txt") as f:
        Y = [line.strip() for line in f.readlines()]
    return X, Y

# === One-hot encoding das letras A-Z ===
def one_hot_encode_label(label):
    vec = [0] * 26
    index = ord(label.upper()) - ord('A')
    vec[index] = 1
    return vec

# === Softmax e sua derivada ===
def softmax(z):
    exps = [pow(2.718281828459, x) for x in z]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# === Funções de ativação ===
def relu(x):
    return [max(0, v) for v in x]

def relu_derivative(x):
    return [1 if v > 0 else 0 for v in x]

# === Inicialização dos pesos ===
def initialize_weights(input_size, hidden_size, output_size):
    import random
    W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
    b1 = [0] * hidden_size
    W2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
    b2 = [0] * output_size
    return W1, b1, W2, b2

# === Produto escalar ===
def dot_product(weights, inputs):
    return [sum(w * x for w, x in zip(weight_row, inputs)) for weight_row in weights]

# === Feedforward ===
def forward_pass(x, W1, b1, W2, b2):
    z1 = [sum(w * xi for w, xi in zip(w_row, x)) + b for w_row, b in zip(W1, b1)]
    a1 = relu(z1)
    z2 = [sum(w * ai for w, ai in zip(w_row, a1)) + b for w_row, b in zip(W2, b2)]
    a2 = softmax(z2)
    return z1, a1, z2, a2

# === Treinamento com backpropagation ===
def train(X, Y, hidden_size=32, epochs=10, learning_rate=0.01):
    input_size = len(X[0])
    output_size = 26  # Letras A-Z
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        correct = 0
        for x, label in zip(X, Y):
            y_true = one_hot_encode_label(label)

            # Forward
            z1, a1, z2, y_pred = forward_pass(x, W1, b1, W2, b2)

            # Acurácia
            if y_pred.index(max(y_pred)) == y_true.index(1):
                correct += 1

            # Backpropagation
            dL_dz2 = [y_pred[i] - y_true[i] for i in range(output_size)]
            dL_dW2 = [[dL_dz2[i] * a1[j] for j in range(hidden_size)] for i in range(output_size)]
            dL_db2 = dL_dz2[:]

            dL_da1 = [sum(dL_dz2[k] * W2[k][i] for k in range(output_size)) for i in range(hidden_size)]
            dL_dz1 = [dL_da1[i] * relu_derivative([z1[i]])[0] for i in range(hidden_size)]
            dL_dW1 = [[dL_dz1[i] * x[j] for j in range(input_size)] for i in range(hidden_size)]
            dL_db1 = dL_dz1[:]

            # Atualização dos pesos
            for i in range(output_size):
                for j in range(hidden_size):
                    W2[i][j] -= learning_rate * dL_dW2[i][j]
            for i in range(output_size):
                b2[i] -= learning_rate * dL_db2[i]

            for i in range(hidden_size):
                for j in range(input_size):
                    W1[i][j] -= learning_rate * dL_dW1[i][j]
            for i in range(hidden_size):
                b1[i] -= learning_rate * dL_db1[i]

        acc = correct / len(X)
        print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")

    return W1, b1, W2, b2

# === Avaliação ===
def evaluate(X, Y, W1, b1, W2, b2):
    correct = 0
    for x, label in zip(X, Y):
        _, _, _, y_pred = forward_pass(x, W1, b1, W2, b2)
        pred_index = y_pred.index(max(y_pred))
        true_index = ord(label.upper()) - ord('A')
        if pred_index == true_index:
            correct += 1
    acc = correct / len(X)
    print(f"Final Accuracy: {acc:.4f}")

# === Execução ===
if __name__ == "__main__":
    X, Y = load_data()
    W1, b1, W2, b2 = train(X, Y, hidden_size=32, epochs=20, learning_rate=0.05)
    evaluate(X, Y, W1, b1, W2, b2)
