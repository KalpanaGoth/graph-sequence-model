import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

nltk.download('punkt', quiet=True)

class GraphRepresentation:
    def __init__(self, max_nodes=100):
        self.max_nodes = max_nodes
    
    def text_to_graph(self, text):
        tokens = word_tokenize(str(text).lower())[:self.max_nodes]
        num_nodes = len(tokens)
        edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
        return tokens, edge_index

class EdgeWeightCalculator(nn.Module):
    def __init__(self, embedding_dim):
        super(EdgeWeightCalculator, self).__init__()
        self.weight_nn = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j], dim=-1)
        return self.weight_nn(edge_features)

class GraphBasedSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.5):
        super(GraphBasedSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.edge_weight_calculator = EdgeWeightCalculator(embedding_dim)
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.dropout(x)
        
        row, col = edge_index
        edge_weight = self.edge_weight_calculator(x[row], x[col]).squeeze()
        
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index, edge_weight))
        x = self.dropout(x)
        
        x = x.mean(dim=0)
        return torch.sigmoid(self.fc(x))

def prepare_data(data_path, max_nodes=100):
    df = pd.read_csv(data_path)
    print("DataFrame columns:", df.columns)
    print("First few rows of the DataFrame:")
    print(df.head())
    
    text_column = None
    sentiment_column = None
    for col in df.columns:
        if 'text' in col.lower() or 'review' in col.lower():
            text_column = col
        elif 'sentiment' in col.lower() or 'label' in col.lower():
            sentiment_column = col
    
    if text_column is None or sentiment_column is None:
        raise ValueError("Could not identify text and sentiment columns. Please check your CSV file.")
    
    print(f"Using '{text_column}' as text column and '{sentiment_column}' as sentiment column.")
    
    graph_repr = GraphRepresentation(max_nodes)
    
    graphs = []
    labels = []
    word_to_index = {'<PAD>': 0}
    current_index = 1
    
    for _, row in df.iterrows():
        tokens, edge_index = graph_repr.text_to_graph(row[text_column])
        node_indices = [word_to_index.setdefault(token, len(word_to_index)) for token in tokens]
        node_indices = node_indices[:max_nodes] + [0] * (max_nodes - len(node_indices))
        
        graphs.append(Data(x=torch.tensor(node_indices), edge_index=edge_index))
        labels.append(float(row[sentiment_column]))
    
    return graphs, labels, len(word_to_index)

def train(model, train_graphs, train_labels, optimizer, criterion):
    model.train()
    total_loss = 0
    for graph, label in zip(train_graphs, train_labels):
        optimizer.zero_grad()
        output = model(graph.x, graph.edge_index)
        loss = criterion(output, torch.tensor([label], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_graphs)

def evaluate(model, graphs, labels, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    with torch.no_grad():
        for graph, label in zip(graphs, labels):
            output = model(graph.x, graph.edge_index)
            loss = criterion(output, torch.tensor([label], dtype=torch.float32))
            total_loss += loss.item()
            predictions.append(output.item())
    accuracy = accuracy_score(labels, [1 if p > 0.5 else 0 for p in predictions])
    return total_loss / len(graphs), accuracy

def run_experiment(data_path, max_nodes=100, embedding_dim=64, hidden_dim=32, epochs=10, l2_reg=1e-5):
    graphs, labels, vocab_size = prepare_data(data_path, max_nodes)
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.2, random_state=42)
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(train_graphs, train_labels, test_size=0.2, random_state=42)
    
    model = GraphBasedSentimentModel(vocab_size, embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2_reg)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train(model, train_graphs, train_labels, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_graphs, val_labels, criterion)
        test_loss, test_acc = evaluate(model, test_graphs, test_labels, criterion)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    return model

if __name__ == "__main__":
    data_path = "data/imdb_sample.csv"  # Replace with any path to data
    try:
        model = run_experiment(data_path)
        print("Experiment completed!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your CSV file and ensure it contains appropriate text and sentiment columns.")