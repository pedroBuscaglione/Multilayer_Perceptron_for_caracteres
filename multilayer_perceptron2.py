# === Carregamento dos dados ===

import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def initialize_parameters(input_size, hidden_size, output_size):
    def random_matrix(rows, cols):
        return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

    def zero_vector(size):
        return [0.0 for _ in range(size)]

    W1 = random_matrix(input_size, hidden_size)
    b1 = zero_vector(hidden_size)
    W2 = random_matrix(hidden_size, output_size)
    b2 = zero_vector(output_size)

    return W1, b1, W2, b2

def load_data():
    with open("X.txt") as f:
        X = []
        for line in f:
            # Remove espaços extras e separa corretamente
            numbers = line.strip().replace(',', ' ').split()
            # Converte para int
            X.append([int(n) for n in numbers])
    
    with open("Y_letra.txt") as f:
        Y = [line.strip().upper() for line in f.readlines()]

    print("Exemplo de X:", X[0])
    print("Tipo de cada elemento:", type(X[0][0]))
    
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
def forward_pass(X, W1, b1, W2, b2):
    Z1 = []
    A1 = []
    for x in X:
        # Camada oculta
        z1 = [sum(w * xi for w, xi in zip(w_row, x)) + b for w_row, b in zip(W1, b1)]
        a1 = [sigmoid(z) for z in z1]
        Z1.append(z1)
        A1.append(a1)

    Z2 = []
    A2 = []
    for a1 in A1:
        z2 = [sum(w * ai for w, ai in zip(w_row, a1)) + b for w_row, b in zip(W2, b2)]
        a2 = softmax(z2)
        Z2.append(z2)
        A2.append(a2)

    return Z1, A1, Z2, A2

def one_hot_encode(Y, num_classes):
    one_hot = []
    for label in Y:
        vector = [0] * num_classes
        index = ord(label.upper()) - ord('A')  # Assume letras de A a Z
        vector[index] = 1
        one_hot.append(vector)
    return one_hot

# === Treinamento com backpropagation ===
def train(X, Y, hidden_size=32, epochs=50, learning_rate=0.05, patience=5):
    input_size = len(X[0])
    output_size = len(set(Y))
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    Y_encoded = one_hot_encode(Y, output_size)

    best_accuracy = 0
    epochs_without_improvement = 0
    accuracy_history = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)
        loss = cross_entropy_loss(Y_encoded, A2)
        accuracy = calculate_accuracy(Y, A2)
        accuracy_history.append(accuracy)
        
        print(f"Epoch {epoch+1:2d} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n⏹️ Early stopping: accuracy didn’t improve for {patience} epochs.")
                break

        # Backpropagation
        dW1, db1, dW2, db2 = backward_pass(X, Y_encoded, Z1, A1, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    # Print visual graph in terminal
    print("\n📈 Accuracy over epochs:")
    for i, acc in enumerate(accuracy_history):
        bar = "#" * int(acc * 50)
        print(f"Epoch {i+1:2d}: [{bar:<50}] {acc:.4f}")

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
