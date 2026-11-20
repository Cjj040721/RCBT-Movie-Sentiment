import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = batch["labels"].size(0)
        total_loss += output.loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels_all = []
    predictions_all = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            predictions = output.logits.argmax(dim=-1)
            labels_all.extend(batch["labels"].cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())
            batch_size = batch["labels"].size(0)
            total_loss += output.loss.item() * batch_size
            total_examples += batch_size
    labels = np.asarray(labels_all)
    predictions = np.asarray(predictions_all)
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }
