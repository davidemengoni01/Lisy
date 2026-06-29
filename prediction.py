import numpy as np
import torch

from model.lisyNet import LISYNet


# ===========================
# Carica labels
# ===========================

data = np.load("dataset\\lisy_dataset.npz", allow_pickle=True)

labels_map = data["labels_map"].item()

idx_to_label = {
    v: k
    for k, v in labels_map.items()
}


# ===========================
# Carica modello
# ===========================

model = LISYNet(len(labels_map))

model.load_state_dict(torch.load("model\\LISYNet.pth"))

model.eval()


# ===========================
# Funzione di predizione
# ===========================

def predict(keypoints):

    keypoints = np.array(
        keypoints,
        dtype=np.float32
    ).reshape(1, 63)

    tensor = torch.tensor(keypoints)

    with torch.no_grad():

        output = model(tensor)

        prediction = output.argmax(dim=1).item()

    return idx_to_label[prediction]