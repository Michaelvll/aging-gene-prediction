from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_data, get_balanced_data, normalize_features
import wandb
import optuna
from optuna.integration import WeightsAndBiasesCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cpu':
    print("WARNING: CUDA is not available. Running on CPU may be slow.")

wandb.init(project='gene')

data = load_data()
X_balanced, y_balanced = get_balanced_data(data)
print(len(X_balanced['mcg']), len(X_balanced['atac']))

OUTPUT_DIM = 3  # number of classes (-1, 0, 1)
NUM_EPOCHS = 200

class TwoHeadTransformerModel(nn.Module):
    def __init__(self, mcg_input_dim, atac_input_dim, hidden_dim, output_dim, num_layers=2, num_heads=1, dropout=0.1):
        super(TwoHeadTransformerModel, self).__init__()
        self.mcg_embedding = nn.Linear(mcg_input_dim, hidden_dim)
        self.atac_embedding = nn.Linear(atac_input_dim, hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
        self.mcg_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.atac_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, mcg_x, mcg_mask, atac_x, atac_mask):
        mcg_x = self.mcg_embedding(mcg_x)
        atac_x = self.atac_embedding(atac_x)
        
        mcg_x = self.mcg_transformer(mcg_x, src_key_padding_mask=~mcg_mask.bool())
        atac_x = self.atac_transformer(atac_x, src_key_padding_mask=~atac_mask.bool())
        
        # Global average pooling
        mcg_x = mcg_x.mean(dim=1)
        atac_x = atac_x.mean(dim=1)
        
        # Concatenate MCG and ATAC embeddings
        combined_x = torch.cat((mcg_x, atac_x), dim=1)
        
        output = self.classifier(combined_x)
        return output

class CombinedGeneDataset(Dataset):
    def __init__(self, mcg_data, atac_data, labels):
        self.mcg_data = mcg_data
        self.atac_data = atac_data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mcg_gene_data = torch.FloatTensor(self.mcg_data[idx])
        atac_gene_data = torch.FloatTensor(self.atac_data[idx])
        label = torch.LongTensor([self.labels[idx] + 1])  # Add 1 to shift labels to 0, 1, 2
        mcg_mask = torch.ones(len(mcg_gene_data))
        atac_mask = torch.ones(len(atac_gene_data))
        return mcg_gene_data, atac_gene_data, label, mcg_mask, atac_mask

def combined_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    mcg_sequences, atac_sequences, labels, mcg_masks, atac_masks = zip(*batch)
    
    mcg_lengths = [len(seq) for seq in mcg_sequences]
    atac_lengths = [len(seq) for seq in atac_sequences]
    mcg_max_len = max(mcg_lengths)
    atac_max_len = max(atac_lengths)
    
    padded_mcg_seqs = torch.zeros(len(mcg_sequences), mcg_max_len, mcg_sequences[0].size(1))
    padded_atac_seqs = torch.zeros(len(atac_sequences), atac_max_len, atac_sequences[0].size(1))
    padded_mcg_masks = torch.zeros(len(mcg_sequences), mcg_max_len)
    padded_atac_masks = torch.zeros(len(atac_sequences), atac_max_len)
    
    for i, (mcg_seq, atac_seq, mcg_length, atac_length) in enumerate(zip(mcg_sequences, atac_sequences, mcg_lengths, atac_lengths)):
        padded_mcg_seqs[i, :mcg_length] = mcg_seq
        padded_atac_seqs[i, :atac_length] = atac_seq
        padded_mcg_masks[i, :mcg_length] = 1
        padded_atac_masks[i, :atac_length] = 1
    
    return padded_mcg_seqs, padded_atac_seqs, torch.cat(labels), padded_mcg_masks, padded_atac_masks

class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, device, **kwargs):
        super(DeviceDataLoader, self).__init__(dataset, **kwargs)
        self.device = device
        if torch.cuda.is_available() and device.type == 'cuda':
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def __iter__(self):
        self.iter = super(DeviceDataLoader, self).__iter__()
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.batch = [b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in self.batch]
        else:
            self.batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in self.batch]

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

def train_combined_model(X_train_mcg, X_train_atac, y_train, X_test_mcg, X_test_atac, y_test, exp_name, model, lr, batch_size):
    wandb.init(project='gene', group=exp_name)
    
    train_dataset = CombinedGeneDataset(X_train_mcg, X_train_atac, y_train)
    test_dataset = CombinedGeneDataset(X_test_mcg, X_test_atac, y_test)
    train_loader = DeviceDataLoader(train_dataset, device, batch_size=batch_size, shuffle=True, collate_fn=combined_collate_fn, pin_memory=True)
    test_loader = DeviceDataLoader(test_dataset, device, batch_size=batch_size, shuffle=False, collate_fn=combined_collate_fn, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0001)

    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for mcg_x, atac_x, batch_y, mcg_mask, atac_mask in train_loader:
            optimizer.zero_grad()
            outputs = model(mcg_x, mcg_mask, atac_x, atac_mask)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == batch_y.squeeze()).sum().item()
            train_total += batch_y.size(0)
        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for mcg_x, atac_x, batch_y, mcg_mask, atac_mask in test_loader:
                outputs = model(mcg_x, mcg_mask, atac_x, atac_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.squeeze()).sum().item()
        
        accuracy = correct / total
        wandb.log({'epoch': epoch, 'LR': optimizer.param_groups[0]['lr'], 'train_loss': total_loss/len(train_loader), 'train_accuracy': train_correct/train_total, 'test_accuracy': accuracy})

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for mcg_x, atac_x, batch_y, mcg_mask, atac_mask in test_loader:
            outputs = model(mcg_x, mcg_mask, atac_x, atac_mask)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    final_accuracy = sum(np.array(all_predictions) == np.array(all_labels).squeeze()) / len(all_labels)
    print(f'Final Test Accuracy: {final_accuracy:.4f}')
    return final_accuracy

def objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    model = TwoHeadTransformerModel(mcg_input_dim, atac_input_dim, hidden_dim, OUTPUT_DIM, 
                                    num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model = model.to(device)

    accuracy = train_combined_model(X_train_mcg_normalized, X_train_atac_normalized, y_train, 
                                    X_test_mcg_normalized, X_test_atac_normalized, y_test, 
                                    exp_name=f"trial_{trial.number}", model=model, lr=lr, batch_size=batch_size)

    return accuracy

if __name__ == "__main__":
    kf = KFold(n_splits=5, shuffle=True, random_state=25)
    train_index, test_index = next(kf.split(X_balanced['mcg']))  # Use only one fold for hyperparameter search

    X_train_mcg, X_test_mcg = [X_balanced['mcg'][i] for i in train_index], [X_balanced['mcg'][i] for i in test_index]
    X_train_atac, X_test_atac = [X_balanced['atac'][i] for i in train_index], [X_balanced['atac'][i] for i in test_index]
    y_train, y_test = [y_balanced[i] for i in train_index], [y_balanced[i] for i in test_index]

    X_train_mcg_normalized, X_test_mcg_normalized = normalize_features(X_train_mcg, X_test_mcg)
    X_train_atac_normalized, X_test_atac_normalized = normalize_features(X_train_atac, X_test_atac)

    mcg_input_dim = len(X_train_mcg_normalized[0][0])
    atac_input_dim = len(X_train_atac_normalized[0][0])

    wandb_callback = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs={"project": "gene_hyperparameter_search"})
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=500, callbacks=[wandb_callback], n_jobs=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train the final model with the best hyperparameters
    best_params = trial.params
    final_model = TwoHeadTransformerModel(mcg_input_dim, atac_input_dim, best_params['hidden_dim'], OUTPUT_DIM, 
                                          num_layers=best_params['num_layers'], num_heads=best_params['num_heads'], 
                                          dropout=best_params['dropout'])
    final_model = final_model.to(device)

    # Train and evaluate the final model using cross-validation
    cv_accuracies = []
    for train_index, test_index in kf.split(X_balanced['mcg']):
        X_train_mcg, X_test_mcg = [X_balanced['mcg'][i] for i in train_index], [X_balanced['mcg'][i] for i in test_index]
        X_train_atac, X_test_atac = [X_balanced['atac'][i] for i in train_index], [X_balanced['atac'][i] for i in test_index]
        y_train, y_test = [y_balanced[i] for i in train_index], [y_balanced[i] for i in test_index]

        X_train_mcg_normalized, X_test_mcg_normalized = normalize_features(X_train_mcg, X_test_mcg)
        X_train_atac_normalized, X_test_atac_normalized = normalize_features(X_train_atac, X_test_atac)

        accuracy = train_combined_model(X_train_mcg_normalized, X_train_atac_normalized, y_train, 
                                        X_test_mcg_normalized, X_test_atac_normalized, y_test, 
                                        exp_name="final_model", model=final_model, lr=best_params['lr'], batch_size=best_params['batch_size'])
        cv_accuracies.append(accuracy)

    print(f'Mean CV Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}')

