from tqdm import tqdm
import sys
import logging
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
import time
import sqlite3

import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from data_loader import load_data, get_balanced_data, normalize_features

# Load and prepare data
data = load_data()
X_balanced, y_balanced = get_balanced_data(data)
FEATURE_TYPES = ['mcg', 'atac', 'hic', 'genebody']
for k, v in X_balanced.items():
    print(k, len(v))

class FourHeadTransformerModel(nn.Module):
    def __init__(self, mcg_input_dim, atac_input_dim, hic_input_dim, genebody_input_dim, hidden_dim, output_dim, num_layers=2, num_heads=1, dropout=0.1):
        super(FourHeadTransformerModel, self).__init__()
        self.mcg_embedding = nn.Linear(mcg_input_dim, hidden_dim)
        self.atac_embedding = nn.Linear(atac_input_dim, hidden_dim)
        self.hic_embedding = nn.Linear(hic_input_dim, hidden_dim)
        self.genebody_embedding = nn.Linear(genebody_input_dim, hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.mcg_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.atac_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.hic_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.genebody_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Linear(hidden_dim * 4, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, mcg_x, mcg_mask, atac_x, atac_mask, hic_x, hic_mask, genebody_x, genebody_mask):
        mcg_x = self.mcg_embedding(mcg_x)
        atac_x = self.atac_embedding(atac_x)
        hic_x = self.hic_embedding(hic_x)
        genebody_x = self.genebody_embedding(genebody_x)
        
        mcg_x = self.mcg_transformer(mcg_x, src_key_padding_mask=~mcg_mask.bool())
        atac_x = self.atac_transformer(atac_x, src_key_padding_mask=~atac_mask.bool())
        hic_x = self.hic_transformer(hic_x, src_key_padding_mask=~hic_mask.bool())
        genebody_x = self.genebody_transformer(genebody_x, src_key_padding_mask=~genebody_mask.bool())
        
        mcg_x = mcg_x.mean(dim=1)
        atac_x = atac_x.mean(dim=1)
        hic_x = hic_x.mean(dim=1)
        genebody_x = genebody_x.mean(dim=1)

        combined_x = torch.cat((mcg_x, atac_x, hic_x, genebody_x), dim=1)
        
        output = self.classifier(combined_x)
        return output

class CombinedGeneDataset(Dataset):
    def __init__(self, mcg_data, atac_data, hic_data, genebody_data, labels):
        self.mcg_data = mcg_data
        self.atac_data = atac_data
        self.hic_data = hic_data
        self.genebody_data = genebody_data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mcg_gene_data = torch.FloatTensor(self.mcg_data[idx])
        atac_gene_data = torch.FloatTensor(self.atac_data[idx])
        hic_gene_data = torch.FloatTensor(self.hic_data[idx])
        genebody_gene_data = torch.FloatTensor(self.genebody_data[idx])
        label = torch.LongTensor([self.labels[idx] + 1])  # Add 1 to shift labels to 0, 1, 2
        mcg_mask = torch.ones(len(mcg_gene_data))
        atac_mask = torch.ones(len(atac_gene_data))
        hic_mask = torch.ones(len(hic_gene_data))
        genebody_mask = torch.ones(len(genebody_gene_data))
        return mcg_gene_data, atac_gene_data, hic_gene_data, genebody_gene_data, label, mcg_mask, atac_mask, hic_mask, genebody_mask

def combined_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    mcg_sequences, atac_sequences, hic_sequences, genebody_sequences, labels, mcg_masks, atac_masks, hic_masks, genebody_masks = zip(*batch)
    
    mcg_lengths = [len(seq) for seq in mcg_sequences]
    atac_lengths = [len(seq) for seq in atac_sequences]
    hic_lengths = [len(seq) for seq in hic_sequences]
    genebody_lengths = [len(seq) for seq in genebody_sequences]
    mcg_max_len = max(mcg_lengths)
    atac_max_len = max(atac_lengths)
    hic_max_len = max(hic_lengths)
    genebody_max_len = max(genebody_lengths)
    
    padded_mcg_seqs = torch.zeros(len(mcg_sequences), mcg_max_len, mcg_sequences[0].size(1))
    padded_atac_seqs = torch.zeros(len(atac_sequences), atac_max_len, atac_sequences[0].size(1))
    padded_hic_seqs = torch.zeros(len(hic_sequences), hic_max_len, hic_sequences[0].size(1))
    padded_genebody_seqs = torch.zeros(len(genebody_sequences), genebody_max_len, genebody_sequences[0].size(1))
    padded_mcg_masks = torch.zeros(len(mcg_sequences), mcg_max_len)
    padded_atac_masks = torch.zeros(len(atac_sequences), atac_max_len)
    padded_hic_masks = torch.zeros(len(hic_sequences), hic_max_len)
    padded_genebody_masks = torch.zeros(len(genebody_sequences), genebody_max_len)
    
    for i, (mcg_seq, atac_seq, hic_seq, genebody_seq, mcg_length, atac_length, hic_length, genebody_length) in enumerate(zip(mcg_sequences, atac_sequences, hic_sequences, genebody_sequences, mcg_lengths, atac_lengths, hic_lengths, genebody_lengths)):
        padded_mcg_seqs[i, :mcg_length] = mcg_seq
        padded_atac_seqs[i, :atac_length] = atac_seq
        padded_hic_seqs[i, :hic_length] = hic_seq
        padded_genebody_seqs[i, :genebody_length] = genebody_seq
        padded_mcg_masks[i, :mcg_length] = 1
        padded_atac_masks[i, :atac_length] = 1
        padded_hic_masks[i, :hic_length] = 1
        padded_genebody_masks[i, :genebody_length] = 1
    
    return padded_mcg_seqs, padded_atac_seqs, padded_hic_seqs, padded_genebody_seqs, torch.cat(labels), padded_mcg_masks, padded_atac_masks, padded_hic_masks, padded_genebody_masks

def train_combined_model(X_train_mcg, X_train_atac, X_train_hic, X_train_genebody, y_train, 
                         X_val_mcg, X_val_atac, X_val_hic, X_val_genebody, y_val, 
                         hidden_dim, num_layers, num_heads, dropout, lr, batch_size, num_epochs):
    mcg_input_dim = len(X_train_mcg[0][0])
    atac_input_dim = len(X_train_atac[0][0])
    hic_input_dim = len(X_train_hic[0][0])
    genebody_input_dim = len(X_train_genebody[0][0])
    
    train_dataset = CombinedGeneDataset(X_train_mcg, X_train_atac, X_train_hic, X_train_genebody, y_train)
    val_dataset = CombinedGeneDataset(X_val_mcg, X_val_atac, X_val_hic, X_val_genebody, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=combined_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=combined_collate_fn)

    model = FourHeadTransformerModel(mcg_input_dim, atac_input_dim, hic_input_dim, genebody_input_dim, hidden_dim, 3, num_layers=num_layers, num_heads=num_heads, dropout=dropout)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for mcg_x, atac_x, hic_x, genebody_x, batch_y, mcg_mask, atac_mask, hic_mask, genebody_mask in train_loader:
            optimizer.zero_grad()
            outputs = model(mcg_x, mcg_mask, atac_x, atac_mask, hic_x, hic_mask, genebody_x, genebody_mask)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for mcg_x, atac_x, hic_x, genebody_x, batch_y, mcg_mask, atac_mask, hic_mask, genebody_mask in val_loader:
                outputs = model(mcg_x, mcg_mask, atac_x, atac_mask, hic_x, hic_mask, genebody_x, genebody_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.squeeze()).sum().item()
        
        val_accuracy = correct / total
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy

def objective(trial: Trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = 20  # Fixed for faster trials

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_index, val_index in kf.split(X_balanced['mcg']):
        X_train_mcg, X_val_mcg = [X_balanced['mcg'][i] for i in train_index], [X_balanced['mcg'][i] for i in val_index]
        X_train_atac, X_val_atac = [X_balanced['atac'][i] for i in train_index], [X_balanced['atac'][i] for i in val_index]
        X_train_hic, X_val_hic = [X_balanced['hic'][i] for i in train_index], [X_balanced['hic'][i] for i in val_index]
        X_train_genebody, X_val_genebody = [X_balanced['genebody'][i] for i in train_index], [X_balanced['genebody'][i] for i in val_index]
        y_train, y_val = [y_balanced[i] for i in train_index], [y_balanced[i] for i in val_index]
        
        X_train_mcg_normalized, X_val_mcg_normalized = normalize_features(X_train_mcg, X_val_mcg)
        X_train_atac_normalized, X_val_atac_normalized = normalize_features(X_train_atac, X_val_atac)
        X_train_hic_normalized, X_val_hic_normalized = normalize_features(X_train_hic, X_val_hic)
        X_train_genebody_normalized, X_val_genebody_normalized = normalize_features(X_train_genebody, X_val_genebody)
        
        cv_scores.append(train_combined_model(
            X_train_mcg_normalized, X_train_atac_normalized, X_train_hic_normalized, X_train_genebody_normalized, y_train,
            X_val_mcg_normalized, X_val_atac_normalized, X_val_hic_normalized, X_val_genebody_normalized, y_val,
            hidden_dim, num_layers, num_heads, dropout, lr, batch_size, num_epochs
        ))

    mean_cv_score = np.mean(cv_scores)
    wandb.log({
        "trial_number": trial.number,
        "mean_accuracy": mean_cv_score,
        "hidden_dim": trial.params['hidden_dim'],
        "num_layers": trial.params['num_layers'],
        "num_heads": trial.params['num_heads'],
        "dropout": trial.params['dropout'],
        "lr": trial.params['lr'],
        "batch_size": trial.params['batch_size']
    })
    return mean_cv_score

if __name__ == "__main__":
    study_name = f"four_head_transformer_{time.strftime('%Y%m%d-%H%M%S')}"
    storage_name = f"sqlite:///{study_name}.db"
    # Initialize wandb
    wandb.init(project="four_head_transformer", name=study_name)
    
    # Create the WeightsAndBiasesCallback
    wandb_callback = WeightsAndBiasesCallback(metric_name="mean_accuracy", wandb_kwargs={})

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[wandb_callback])  # Adjust the number of trials as needed

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # Train the final model with the best hyperparameters
    best_params = study.best_params
    print("Training final model with best parameters:")
    print(best_params)

    # Prepare the full dataset
    X_mcg_normalized, _ = normalize_features(X_balanced['mcg'], [])
    X_atac_normalized, _ = normalize_features(X_balanced['atac'], [])
    X_hic_normalized, _ = normalize_features(X_balanced['hic'], [])
    X_genebody_normalized, _ = normalize_features(X_balanced['genebody'], [])

    # Train the final model
    final_accuracy = train_combined_model(
        X_mcg_normalized, X_atac_normalized, X_hic_normalized, X_genebody_normalized, y_balanced,
        X_mcg_normalized, X_atac_normalized, X_hic_normalized, X_genebody_normalized, y_balanced,
        best_params['hidden_dim'], best_params['num_layers'], best_params['num_heads'],
        best_params['dropout'], best_params['lr'], best_params['batch_size'], num_epochs=50  # Increase epochs for final training
    )

    print(f"Final model accuracy: {final_accuracy:.4f}")

    # Save study results to CSV
    df = study.trials_dataframe()
    df.to_csv(f"{study_name}_results.csv", index=False)

    wandb.finish()
