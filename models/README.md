# Fine-tuned model artifacts

After running fine-tuning:

```powershell
python scripts/finetune_dronewaste.py --quick
```

LoRA weights are saved to:

```text
models/dronewaste-smolvlm-lora/
```

The runtime loads this automatically when `adapter_config.json` exists, or when:

```powershell
$env:FINETUNED_VLM_PATH = "models/dronewaste-smolvlm-lora"
```

Files:
- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — trained LoRA weights
- `training_manifest.json` — training metadata
- Processor tokenizer files from HuggingFace SmolVLM
