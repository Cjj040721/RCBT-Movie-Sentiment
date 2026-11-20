import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .model import RCBTConfig, RCBTForSequenceClassification


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment inference with a trained RCBT checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = RCBTConfig.from_dict(checkpoint["model_config"])
    encoder = AutoModel.from_pretrained(checkpoint["encoder_name"])
    model = RCBTForSequenceClassification(encoder, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent / "tokenizer")
    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    prediction = int(probabilities.argmax().item())
    print(json.dumps({
        "label": "positive" if prediction == 1 else "negative",
        "negative_probability": round(float(probabilities[0]), 6),
        "positive_probability": round(float(probabilities[1]), 6),
    }, indent=2))


if __name__ == "__main__":
    main()
