import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

from model.lisyNet import LISYNet


# ==========================================================
# CARICAMENTO DATASET
# ==========================================================

data = np.load("lisy_dataset.npz", allow_pickle=True)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

labels_map = data["labels_map"].item()

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", labels_map)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================================
# PARAMETRI
# ==========================================================

num_classes = len(labels_map)

epochs = 50

batch_size = 32

learning_rate = 0.001

patience = 5


# ==========================================================
# K-FOLD
# ==========================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []

best_global_accuracy = 0
best_global_weights = None


# ==========================================================
# TRAINING
# ==========================================================

for fold, (train_ids, val_ids) in enumerate(kfold.split(X), start=1):

    print("\n" + "=" * 50)
    print(f"FOLD {fold}/5")
    print("=" * 50)

    X_train = torch.tensor(X[train_ids], dtype=torch.float32)
    y_train = torch.tensor(y[train_ids], dtype=torch.long)

    X_val = torch.tensor(X[val_ids], dtype=torch.float32)
    y_val = torch.tensor(y[val_ids], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    model = LISYNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_fold_accuracy = 0

    best_fold_weights = copy.deepcopy(model.state_dict())

    early_counter = 0

    # ======================================================

    for epoch in range(epochs):

        ###########################
        # TRAIN
        ###########################

        model.train()

        train_loss = 0

        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (predictions == y_batch).sum().item()

            train_total += len(y_batch)

        train_accuracy = train_correct / train_total

        ###########################
        # VALIDATION
        ###########################

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == y_batch).sum().item()

                val_total += len(y_batch)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Loss {train_loss:.3f} | "
            f"Train {train_accuracy:.2%} | "
            f"Validation {val_accuracy:.2%}"
        )

        ###########################
        # EARLY STOPPING
        ###########################

        if val_accuracy > best_fold_accuracy:

            best_fold_accuracy = val_accuracy

            best_fold_weights = copy.deepcopy(model.state_dict())

            early_counter = 0

        else:

            early_counter += 1

            if early_counter >= patience:

                print("Early stopping!")

                break

    ###############################
    # FINE FOLD
    ###############################

    model.load_state_dict(best_fold_weights)

    fold_accuracies.append(best_fold_accuracy)

    print(f"Best Fold Accuracy: {best_fold_accuracy:.2%}")

    if best_fold_accuracy > best_global_accuracy:

        best_global_accuracy = best_fold_accuracy

        best_global_weights = copy.deepcopy(best_fold_weights)


# ==========================================================
# RISULTATI
# ==========================================================

print("\n")
print("=" * 50)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 50)

for i, acc in enumerate(fold_accuracies, start=1):
    print(f"Fold {i}: {acc:.2%}")

print("-" * 50)

print(f"Mean Accuracy : {np.mean(fold_accuracies):.2%}")
print(f"Std Accuracy  : {np.std(fold_accuracies):.2%}")

print("=" * 50)


# ==========================================================
# SALVATAGGIO MODELLO MIGLIORE
# ==========================================================

best_model = LISYNet(num_classes)

best_model.load_state_dict(best_global_weights)

torch.save(best_model.state_dict(), "LISYNet.pth")

print("\nBest model saved as LISYNet.pth")