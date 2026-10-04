import numpy as np
import matplotlib.pyplot as plt

def init_params(input_size=784, hidden_size=10, output_size=10):
    """
    Initializes weights with small random values and biases with zeros.
    784 features (28x28 images), 10 hidden neurons, 10 output classes (0-9).
    """
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Subtracting np.max(Z) prevents numerical overflow (standard stability trick)
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Performance Metrics #
def compute_loss(A2, Y):
    """
    Categorical Cross-Entropy Loss.
    Y should be one-hot encoded (shape: 10 x num_examples)
    """
    m = Y.shape[1]
    # Adding a tiny epsilon (1e-15) avoids log(0) errors
    loss = -1/m * np.sum(Y * np.log(A2 + 1e-15))
    return loss

def get_accuracy(A2, Y_labels):
    """
    Compares the highest predicted probability index to the actual true label.
    """
    predictions = np.argmax(A2, axis=0)
    return np.sum(predictions == Y_labels) / Y_labels.size

def ReLU_derivative(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Y must be one-hot encoded (shape: 10 x m)
    m is the number of training examples
    """
    m = X.shape[1]
    
    # --- Output Layer Gradients ---
    # dZ2 is the error at the output layer (Predictions minus Actuals)
    dZ2 = A2 - Y 
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # --- Hidden Layer Gradients ---
    # Backpropagate the error through W2, then multiply by ReLU derivative
    dZ1 = np.dot(W2.T, dZ2) * ReLU_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


# Train Network
def train_network(X, Y, Y_labels, iterations, learning_rate):
    # 1. Initialize parameters
    W1, b1, W2, b2 = init_params(input_size=X.shape[0], hidden_size=10, output_size=10)
    
    # Lists to store metrics for Matplotlib visualization
    loss_history = []
    accuracy_history = []
    
    # 2. Optimization Loop
    for i in range(iterations):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # Calculate performance
        loss = compute_loss(A2, Y)
        accuracy = get_accuracy(A2, Y_labels)
        
        # Save history for visualization
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        
        # Backward propagation
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        
        # Update weights and biases
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        # Print progress every 50 iterations
        if i % 50 == 0 or i == iterations - 1:
            print(f"Iteration {i:03d} -> Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")
            
    return W1, b1, W2, b2, loss_history, accuracy_history


# #####
def load_local_mnist(train_path="mnist_train.csv", test_path="mnist_test.csv"):
    """
    Loads MNIST from local CSV files, normalizes pixels, and splits features/labels.
    Assumes first column is the label.
    """
    print("Loading training data from local file...")
    # delimiter=',' reads comma-separated spreadsheet values
    # skiprows=1 skips the header row (label, pixel1, pixel2...)
    train_data = np.loadtxt(train_path, delimiter=',', skiprows=1)
    
    print("Loading test data from local file...")
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    
    # 1. Isolate the labels (first column)
    Y_train_labels = train_data[:, 0].astype(int)
    Y_test_labels = test_data[:, 0].astype(int)
    
    # 2. Isolate features, normalize pixels to [0, 1], and Transpose 
    # Transposing (.T) flips rows into columns so shape becomes (784, num_examples)
    X_train = (train_data[:, 1:] / 255.0).T
    X_test = (test_data[:, 1:] / 255.0).T
    
    # 3. One-hot encode the target labels for training
    num_train_examples = Y_train_labels.size
    Y_train = np.zeros((10, num_train_examples))
    Y_train[Y_train_labels, np.arange(num_train_examples)] = 1
    
    num_test_examples = Y_test_labels.size
    Y_test = np.zeros((10, num_test_examples))
    Y_test[Y_test_labels, np.arange(num_test_examples)] = 1
    
    print(f"Dataset Loaded Successfully!")
    print(f"-> Train Features Shape: {X_train.shape} | Test Features Shape: {X_test.shape}")
    
    return X_train, Y_train, Y_train_labels, X_test, Y_test, Y_test_labels


def test_network(X_test, Y_test, Y_test_labels, W1, b1, W2, b2):
    """
    Runs forward propagation using optimized parameters on completely unseen data
    to verify generalization accuracy and detect overfitting.
    """
    # Forward pass only - no backpropagation here!
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_test)
    
    test_loss = compute_loss(A2, Y_test)
    test_accuracy = get_accuracy(A2, Y_test_labels)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS (TEST SET)")
    print("="*40)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("="*40)
    
    return test_accuracy


# 1. Create fake data: 5 sample "images" of 784 pixels each
# np.random.seed(42)
# X_dummy = np.random.randn(784, 5) 
# Y_labels_dummy = np.array([1, 3, 9, 4, 1]) # True digit categories

# # One-hot encode the dummy labels
# Y_dummy = np.zeros((10, 5))
# Y_dummy[Y_labels_dummy, np.arange(5)] = 1

# # 2. Run the network
# W1, b1, W2, b2 = init_params()
# Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_dummy)

# Generate synthetic dataset: 100 samples, 784 pixels each
# np.random.seed(42)
# X_train = np.random.randn(784, 100)
# Y_labels_train = np.random.randint(0, 10, size=100)

# # One-hot encode the target labels
# Y_train = np.zeros((10, 100))
# Y_train[Y_labels_train, np.arange(100)] = 1

# # Train for 300 iterations with a learning rate of 0.1
# W1, b1, W2, b2, losses, accuracies = train_network(
#     X_train, Y_train, Y_labels_train, iterations=301, learning_rate=0.1
# )

# 3. Print verification stats
# print("Output Probabilities Shape (Should be 10, 5):", A2.shape)
# print("Initial Loss:", compute_loss(A2, Y_dummy))
# print("Initial Accuracy:", get_accuracy(A2, Y_labels_dummy))

# print("Loss History:", losses)
# print("Accuracy History:", accuracies)


def plot_training_results(loss_history, accuracy_history):
    """
    Plots the Loss and Accuracy curves side-by-side across training iterations.
    """
    iterations = range(len(loss_history))
    
    plt.figure(figsize=(14, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(iterations, loss_history, color='crimson', linewidth=2, label='Training Loss')
    plt.title('Loss vs. Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations (Epochs)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 2. Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracy_history, color='teal', linewidth=2, label='Training Accuracy')
    plt.title('Accuracy vs. Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations (Epochs)', fontsize=12)
    plt.ylabel('Accuracy (Fraction)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)      # time to render the plot



# 1. Load your local dataset
X_train, Y_train, Y_train_labels, X_test, Y_test, Y_test_labels = load_local_mnist(
    train_path="mnist_train.csv", 
    test_path="mnist_test.csv"
)

# 2. Train the network parameters over 300 cycles
print("\nStarting model optimization...")
W1, b1, W2, b2, losses, accuracies = train_network(
    X_train, Y_train, Y_train_labels, iterations=301, learning_rate=0.10
)

# 3. Plot the training behavior to see the convergence
plot_training_results(losses, accuracies)

# 4. Final step: Validate against the Test Set to check for overfitting
test_network(X_test, Y_test, Y_test_labels, W1, b1, W2, b2)


plt.show()