from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from environmental_os.training.dataset import load_training_pair, load_vlm_samples


@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    dataset_json: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 1
    max_samples: int | None = None
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    save_steps: int = 100
    logging_steps: int = 10

    @classmethod
    def from_json(cls, path: str | Path, quick: bool = False) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if quick and "quick_mode" in data:
            q = data["quick_mode"]
            data["max_samples"] = q.get("max_samples", 80)
            data["max_steps"] = q.get("max_steps", 40)
            data["epochs"] = q.get("epochs", 1)
        return cls(
            base_model=data["base_model"],
            output_dir=data["output_dir"],
            dataset_json=data["dataset_json"],
            lora_r=data.get("lora_r", 8),
            lora_alpha=data.get("lora_alpha", 16),
            lora_dropout=data.get("lora_dropout", 0.05),
            learning_rate=data.get("learning_rate", 2e-4),
            epochs=data.get("epochs", 1),
            max_samples=data.get("max_samples"),
            max_steps=data.get("max_steps"),
            batch_size=data.get("batch_size", 1),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 4),
            max_length=data.get("max_length", 512),
            save_steps=data.get("save_steps", 100),
            logging_steps=data.get("logging_steps", 10),
        )


def train_dronewaste_lora(config: TrainConfig) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_linear_schedule_with_warmup

    samples = load_vlm_samples(config.dataset_json, max_samples=config.max_samples)
    if not samples:
        raise RuntimeError(f"No valid training samples in {config.dataset_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(config.base_model)
    model = AutoModelForImageTextToText.from_pretrained(config.base_model, torch_dtype=dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class VLMTrainDataset(Dataset):
        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict:
            image, user_text, target_text = load_training_pair(self.items[idx])
            prompt = f"User: {user_text}\nAssistant: {target_text}"
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.squeeze(0)
            labels = input_ids.clone()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
            }

    def collate(batch: list[dict]) -> dict:
        keys = batch[0].keys()
        out = {}
        for key in keys:
            if batch[0][key] is None:
                out[key] = None
                continue
            out[key] = torch.stack([item[key] for item in batch])
        return out

    dataset = VLMTrainDataset(samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = config.max_steps or (len(loader) * config.epochs // config.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    running_loss = 0.0
    history: list[dict] = []

    for epoch in range(config.epochs):
        for batch in loader:
            batch = {k: v.to(device) if v is not None and hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % config.logging_steps == 0:
                    history.append({"step": global_step, "loss": round(running_loss, 4)})
                    print(f"step={global_step} loss={running_loss:.4f}")
                running_loss = 0.0

            global_step += 1
            if config.max_steps and global_step >= config.max_steps:
                break
        if config.max_steps and global_step >= config.max_steps:
            break

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    manifest = {
        "base_model": config.base_model,
        "output_dir": str(output_dir),
        "samples_trained": len(samples),
        "steps": global_step,
        "device": str(device),
        "history": history[-20:],
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
