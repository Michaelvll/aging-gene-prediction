# Step 3: Train a sequence model on the training set
# Use pytorch to train a LSTM classifier
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Initialize the model
model = LSTMClassifier(input_size=len(FEATURE_NAMES), hidden_size=16, num_classes=3)
# Initialize the weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# Pad sequences to the same length
def pad_sequences(sequences):
    return pad_sequence([torch.FloatTensor(seq) for seq in sequences], batch_first=True)

X_train = pad_sequences(X_train_normalized)
X_test = pad_sequences(X_test_normalized)
y_train = torch.tensor([int(i) + 1 for i in y_train_raw], dtype=torch.long)
y_test = torch.tensor([int(i) + 1 for i in y_test_raw], dtype=torch.long)


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(50):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        train_total += batch_y.size(0)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predictions = test_outputs.argmax(dim=1)
        test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean()
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}, Test Acc: {test_acc:.4f}')
