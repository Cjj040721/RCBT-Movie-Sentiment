import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import load_imdb_dataloaders
from .engine import evaluate, train_one_epoch
from .model import RCBTConfig, RCBTForSequenceClassification


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RCBT on the official IMDB train/test split")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--output", type=Path, default=Path("outputs/rcbt-run"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=320)
    parser.add_argument("--transformer-layers", type=int, default=2)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, validation_loader, test_loader = load_imdb_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )
    config = RCBTConfig(
        cnn_channels=args.cnn_channels,
        lstm_hidden_size=args.lstm_hidden_size,
        transformer_layers=args.transformer_layers,
    )
    model = RCBTForSequenceClassification.from_pretrained(args.model_name, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        history.append(record)
        print(json.dumps(record))
        if validation["f1_macro"] > best_f1:
            best_f1 = validation["f1_macro"]
            torch.save({
                "encoder_name": args.model_name,
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
            }, args.output / "best.pt")
            tokenizer.save_pretrained(args.output / "tokenizer")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, device)
    result = {
        "encoder_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "train_examples": len(train_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "best_validation_f1": best_f1,
        "test": test,
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
