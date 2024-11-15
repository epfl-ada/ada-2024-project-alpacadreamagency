import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split


def get_model(feature_size, genre_size, layer_size):

    """
    Builds a neural network model for multi-label classification of movie genres.
    """

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
    
    # Output layer
    torch.nn.Linear(layer_size, genre_size),
    torch.nn.Sigmoid() # To get probabilities
    )

    return model


def train_model(batches_train, batches_test, model, optimizer, classification_threshold, device):
    """
    Trains a neural network model for multi-label genre classification and evaluates the model on test batches, 
    tracking performance metrics like loss, accuracy, F-score, precision, and recall.
    """


    # Set model to training mode 
    model.train()

    loss_history = []
    accuracy_history = []
    f_score_history = []
    precision_history = []
    recall_history = []
    # Change the loop to get batch_idx, data and target from train_loader
    for sample_i, (data, target) in enumerate(zip(batches_train, batches_test)):
        target = target.squeeze(1)
        N = data.shape[0] 
        Dy = target.shape[1] # Number of possible genre 
        
        # Move the data to the device
        data = data.float().to(device)
        target = target.float().to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = torch.nn.BCELoss()(output, target)
        loss_float = loss.item()
        # Backpropagate loss & Perform an optimizer step
        loss.backward()
        optimizer.step()
        
        
        # Compute accuracy and loss of the batch
        output_hot, correct = get_output_hot(output, target, classification_threshold)
        accuracy = correct.item() / (N * Dy) 
        f_score, precision, recall = compute_avg_f_score(output_hot, target)
        
        accuracy_history.append(accuracy)
        loss_history.append(loss_float)
        f_score_history.append(f_score)
        precision_history.append(precision)
        recall_history.append(recall)
        
        del data
        del target
        
        if sample_i % (1000 // N) == 0: # Every 1000 samples
            print(f'Batch {loss_float = :.4f}')
            print(f'Batch {accuracy = :.4f}')
            print(f'Batch {f_score = :.4f}')
            print(f'Batch {precision = :.4f}')
            print(f'Batch {recall = :.4f}')
    # print(f"{loss_history = }, {accuracy_history = }")
    torch.cuda.empty_cache()
    return model, loss_history, accuracy_history, f_score_history, precision_history, recall_history


def get_output_hot(output, target, classification_threshold):
    """
    Takes output probabilities from the neural network and applies a 
    classification threshold to generate binary predictions (hot-encoded output). 
    It also computes the number of correct predictions by comparing the binary predictions with the target labels.
    """

    output_hot = (output > classification_threshold).int() # we have |N| x |genre| matrix
    correct = torch.sum(output_hot == target)

    return output_hot, correct


def compute_avg_f_score(output_hot, target):
    """
    Calculates the average F-score, precision, and recall for multi-label classification.
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
    Generates batches of training and target data for model training.
    """

    assert len(training_set) == len(target_set), "Lists must be of equal length."
    
    batches_train = []
    batches_test = []
    for i in range(0, len(training_set), batch_size):
        batches_train.append(torch.tensor(training_set.iloc[i:i+batch_size].values, dtype=torch.float64))
        batches_test.append(torch.tensor(target_set.iloc[i:i+batch_size].values.tolist()))
        
    return batches_train, batches_test


def test_model(model, testing_set, testing_target_set, classification_threshold):
    """
    Evaluates the performance of a trained neural network model on a test dataset. Also computes performance metrics and prints the results.
    """

    N = testing_set.shape[0] 
    Dy = testing_target_set.shape[1] 
    
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
    
    return output_hot
    