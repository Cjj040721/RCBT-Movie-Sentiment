import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


def _limit(dataset, limit: int | None, seed: int):
    if limit is not None and limit < len(dataset):
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def load_imdb_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
    seed: int,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    datasets = {
        "train": _limit(split["train"], train_limit, seed),
        "validation": _limit(split["test"], validation_limit, seed + 1),
        "test": _limit(raw["test"], test_limit, seed + 2),
    }

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = batch["label"]
        return encoded

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loaders = {}
    for name, dataset in datasets.items():
        encoded = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        encoded.set_format(type="torch")
        generator = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(
            encoded,
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator,
            collate_fn=collator,
        )
    return loaders["train"], loaders["validation"], loaders["test"]
