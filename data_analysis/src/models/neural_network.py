import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split
from src.utils.settings import Model_Settings

def get_model(feature_size, genre_size, layer_size, dense_shape):
    """
        Get the model factory. Here we build the structure of the neural network.
        Input layer size: feature_size 
        Hiden layers size: layer_size 
        Output layer size: genre_size 
    """
    if dense_shape:
        model = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(feature_size, layer_size),
            torch.nn.ReLU(),
            
            # Hidden layers
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            
            # Output layer
            torch.nn.Linear(layer_size, genre_size),
            torch.nn.Sigmoid() # To get probabilities
        )
    else:
        model = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(feature_size, 64),
            
            # Hidden layers DOWN
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            
            # Hidden layers UP
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            
            # Hidden layers DOWN
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            
            # Output layer
            torch.nn.Linear(64, genre_size),
            torch.nn.Sigmoid() # To get probabilities
        )

    return model

def start_train_model(
    training_columns, training_set, training_target_set, 
    testing_set, testing_target_set,
    model_settings: Model_Settings, print_result = True
):
    NEW_GENRE = pd.read_csv(r"src\utils\categories.csv")
    
    feature_size = len(training_columns)
    genre_size = len(NEW_GENRE["subgenres"])

    model_factory = lambda: get_model(feature_size, genre_size, model_settings.LAYER_SIZE, model_settings.DENSE_SHAPE)
    model = model_factory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    criterion = torch.nn.BCELoss()

    optimizer_kwargs = dict(
        lr=3e-4,
        weight_decay=1e-3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    
    # print(f"{batches_train = }")
    # print(f"{batches_test = }")
    trained_model, histories = train_model(
        model, training_set, training_target_set, testing_set, testing_target_set, optimizer, criterion, 
        model_settings, device
    ) 

    if print_result:
        print_training_results(*histories)
    
    return trained_model, histories

def train_model(model, training_set, training_target_set, testing_set, testing_target_set,
                optimizer, criterion, model_settings: Model_Settings, device):
    """
        Trains a neural network model for multi-label genre classification and evaluates the model on test batches, 
        Get the score over all the training.
        Return the model and the training history scores.
    """
    # Set model to training mode 
    model.train()

    loss_history = []
    accuracy_history = []
    f_score_history = []
    precision_history = []
    recall_history = []
    
    best_val_loss = float('inf')  # To keep track of the best validation loss
    epochs_without_improvement = 0
    
    for epoch in range(model_settings.EPOCHS):
        print(f'Epoch = {epoch}')
        data_loader = get_training_batch(training_set, training_target_set)
        
        accuracy, loss_float, f_score, precision, recall = train_epoch(model, data_loader, device, optimizer, criterion, model_settings)
        
        accuracy_history.append(accuracy)
        loss_history.append(loss_float)
        f_score_history.append(f_score)
        precision_history.append(precision)
        recall_history.append(recall)
        print(f' · {accuracy = :.4f}')
        print(f' · {f_score = :.4f}')
        print(f' · {precision = :.4f}')
        print(f' · {recall = :.4f}')
                
        data_loader_validation = get_training_batch(testing_set, testing_target_set)
        val_loss, validation_samples = validate_model(model, data_loader_validation, criterion, device)
        print(f" · Validation loss ({validation_samples} samples): {val_loss:.4f}")
        
        if val_loss < best_val_loss - model_settings.PATIENCE_EPS:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement +=1 
        
        if epochs_without_improvement >= model_settings.PATIENCE:
            print(f"Early stopping: No improvement in validation loss for {model_settings.PATIENCE} epochs.")
            break
    # print(f"{loss_history = }, {accuracy_history = }")
    torch.cuda.empty_cache()
    return model, (loss_history, accuracy_history, f_score_history, precision_history, recall_history)

def train_epoch(model, data_loader, device, optimizer, criterion, model_settings: Model_Settings):
    for sample_i, (data, target) in enumerate(data_loader):
        target = target.squeeze(1)
        N = data.shape[0] 
        Dy = target.shape[1] # Number of possible genre 
        
        # Move the data to the device
        data = data.float().to(device)
        target = target.float().to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss_float = loss.item()
        # Backpropagate loss & Perform an optimizer step
        loss.backward()
        optimizer.step()
                            
        if sample_i % model_settings.VERBOSE_INTERVAl == 0: # Every 1000 samples
            # Compute accuracy and loss of the batch
            print(f" - Loss = {loss_float}")
            
    output_hot, correct = get_output_hot(output, target, model_settings.THRESHOLD_GENERAL)
    accuracy = correct.item() / (N * Dy) 
    f_score, precision, recall = compute_avg_f_score(output_hot, target)
    return accuracy, loss_float, f_score, precision, recall

def validate_model(model, data_loader_validation, criterion, device):
    """
    Validates the model on the provided validation data.

    This function computes the validation loss over the validation dataset without performing gradient updates.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be validated.
    data_loader_validation : DataLoader
        The DataLoader object for the validation data.
    criterion : Loss
        The loss function to compute the validation loss.

    Returns
    -------
    tuple
        A tuple containing the average validation loss and the number of validation samples.
    """
    model.eval()
    
    loss = 0
    validation_samples = 0
    validation_batches = 0
    with torch.no_grad():  
        for (inputs, targets) in data_loader_validation:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            outputs = model(inputs)
            
            loss += criterion(outputs, targets).item()
            validation_samples += len(targets)
            validation_batches += 1
    
    assert validation_batches > 0, "ERROR: To validate the model at least one sample is needed"
    return loss/(validation_batches), validation_samples






def get_output_hot(output, target, classification_threshold):
    """
        From the predicted values of the NN, get those predicted genre:
            1 if the genre is chosen: > 50%.
            0 o.w:  <= 50%
    """

    output_hot = (output > classification_threshold).int() # we have |N| x |genre| matrix
    correct = torch.sum(output_hot == target)

    return output_hot, correct


def compute_avg_f_score(output_hot, target):
    """
        Compute the Precision, Recall and F-Score for the predictions 'output_hot'.
    """
    target = target.int()
    # print(f"{output_hot = }")
    # print(f"{target = }")
    true_positives = torch.sum((output_hot & target), dim = 1)
    false_positives = torch.sum((output_hot & ~target), dim = 1)
    false_negatives = torch.sum((~output_hot & target), dim = 1)
    
    denominator = true_positives + false_positives
    precision = torch.where(denominator > 0,
                            true_positives / denominator,
                            torch.zeros_like(true_positives)
                            )
    
    denominator = true_positives + false_negatives
    recall = torch.where(denominator > 0,
                         true_positives / denominator,
                         torch.zeros_like(true_positives)
                         )
    
    denominator = precision + recall
    f_score = torch.where(denominator > 0,
                          2*(precision * recall) / denominator,
                          torch.zeros_like(true_positives)
                          )
                          
    return torch.mean(f_score).item(), torch.mean(precision).item(), torch.mean(recall).item()

def get_training_batch(training_set, target_set, batch_size = 10):
    """
    From the split data, yield the batches of 'batch_size' size to train the model: Stochastic training.
    """
    assert len(training_set) == len(target_set), "Lists must be of equal length."
    
    for i in range(0, len(training_set), batch_size):
        batch_train = torch.tensor(training_set.iloc[i:i+batch_size].values, dtype=torch.float64)
        batch_test = torch.tensor(target_set.tolist()[i:i+batch_size], dtype=torch.float64)

        # print(f"{batch_train.shape = }")
        # print(f"{batch_test.shape = }")
        yield batch_train, batch_test
    

def test_model_get_score(trained_model, training_columns, testing_set, testing_target_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    testing_set_t = torch.tensor(testing_set[training_columns].values.tolist(), dtype=torch.float32).to(device)
    testing_target_set_t = torch.tensor(testing_target_set.values.tolist(), dtype=torch.float32).to(device)
    trained_model = trained_model.to(device)
    
    NEW_GENRE = pd.read_csv(r"src\utils\categories.csv")

    print("Testing model...")
    predictions, scores = test_model(trained_model, testing_set_t, testing_target_set_t, Model_Settings.THRESHOLD_GENERAL)

    genre_labels = NEW_GENRE["categories"]
    predicted_genre = []

    print("Preparing output...")
    for movie_prediction in predictions:
        genres = [genre_labels[i] for i, is_genre in enumerate(movie_prediction) if is_genre == 1]
        predicted_genre.append(genres)

    predictins_output = pd.DataFrame({
        'wikipedia_movie_ID': testing_set["wikipedia_movie_ID"],
        'name': testing_set["name"],
        'original_genres': testing_set["new_genres"],
        'predicted_genres': predicted_genre,
    })


    predictins_output.to_csv("movies_predicted_genre.csv", index = False)
    print("DONE!")
    
    return scores


def test_model(model, testing_set, testing_target_set, classification_threshold):
    """
       
        Once the training is done, Evaluates the performance of a trained neural network model on a test dataset:
        use the 'model' to predict the genre of the movies in 'testing_set'.
        Also compute the Precision, Recall and F-Score of it.
    """
    N = testing_set.shape[0] 
    Dy = testing_target_set.shape[1] 
    
    model.eval()
    with torch.no_grad():
        output = model(testing_set)
        
    del testing_set
        
    output_hot, correct = get_output_hot(output, testing_target_set, classification_threshold)
    
    f_score, precision, recall = compute_avg_f_score(output_hot, testing_target_set)
    
    accuracy = correct.item() / (N * Dy) 
    
    print(f'Testing {accuracy = :.4f}')
    print(f'Testing {f_score = :.4f}')
    print(f'Testing {precision = :.4f}')
    print(f'Testing {recall = :.4f}')
    
    return output_hot, [(accuracy, "accuracy"), (f_score, "f_score"), (precision, "precision"), (recall, "recall")]
    
def print_training_results(loss_history, acc_history, f_score_history, precision_history, recall_history):
    batch_indices = [i for i in range(len(loss_history))]
    plt.figure(figsize=(12, 8)) 
    
    plt.plot(batch_indices, loss_history, label="Loss", linestyle="-", marker="o")
    plt.plot(batch_indices, acc_history, label="Accuracy", linestyle="--", marker="x")
    plt.plot(batch_indices, f_score_history, label="F-Score", linestyle="-.", marker="s")
    plt.plot(batch_indices, precision_history, label="Precision", linestyle=":", marker="^")
    plt.plot(batch_indices, recall_history, label="Recall", linestyle="-", marker="d")
    

    # plt.subplot(3, 2, 1)
    plt.xlabel("Batch")
    plt.ylabel("Metric loss")
    plt.title("Training Loss per Batch")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")
    plt.grid(True)

    plt.tight_layout()
    plt.show()        
    
def plot_metrics_bar(metrics):
    """
    Plots a bar chart for the given metrics.
    
    Args:
    metrics: List of tuples where each tuple contains a metric value and its label.
    """
    values, labels = zip(*metrics)  # Unpack metric values and their names
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Performance Metrics")
    plt.ylim(0.0, 1.0)  # Assuming the metrics are normalized between 0 and 1
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)  # Add value labels above bars
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()