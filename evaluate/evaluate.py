# evaluate.py
import torch
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu

def compute_accuracy(predictions, labels):
    """
    Computes the accuracy of the model.
    Args:
    - predictions: The predicted outputs from the model.
    - labels: The actual labels for the data.

    Returns:
    - float: The accuracy of the predictions.
    """
    return accuracy_score(labels, predictions)

def compute_f1_score(predictions, labels):
    """
    Computes the F1 score of the model.
    Args:
    - predictions: The predicted outputs from the model.
    - labels: The actual labels for the data.

    Returns:
    - float: The F1 score of the predictions.
    """
    return f1_score(labels, predictions, average='weighted')

def compute_bleu_score(reference, hypothesis):
    """
    Computes the BLEU score for the predicted sequence.
    Args:
    - reference: The reference (ground truth) sequence.
    - hypothesis: The generated sequence by the model.

    Returns:
    - float: The BLEU score.
    """
    return sentence_bleu([reference], hypothesis)

def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model on the validation or test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): Loss function.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_outputs = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            # Unpack nodes, edges, and labels
            nodes, edges, labels = batch

            # Forward pass
            outputs = model(nodes, edges)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            # Store labels and outputs for further analysis
            all_labels.extend(labels.tolist())
            all_outputs.extend(outputs.tolist())

    # Compute average loss
    avg_loss = total_loss / total_samples

    # Return evaluation metrics
    return {
        'average_loss': avg_loss,
        # Include additional metrics if necessary
    }