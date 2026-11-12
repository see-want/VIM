"""Fine-tune VIM with response-only vision-language LoRA."""

import argparse
from pathlib import Path

from .io import load_config, write_json
from .models import load_model


def train(config, data, output, max_steps=-1):
    model, processor = load_model(config["model"], config, training=True)
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    from .data import IntentDataset

    settings = config["training"]
    train_data = IntentDataset(data / "train.jsonl", config=config["data"])
    val_data = IntentDataset(data / "val.jsonl", "val", config["data"])
    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=config["max_seq_length"],
        resize="max" if config["data"].get("single_frame") else "min",
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    args = SFTConfig(
        output_dir=str(output),
        num_train_epochs=settings["epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation"],
        learning_rate=settings["learning_rate"],
        warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"],
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        eval_steps=settings["eval_steps"],
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=config["data"]["seed"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=settings["workers"],
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config["max_seq_length"],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
    )
    result = trainer.train()
    model.save_pretrained(output / "final_adapter")
    processor.save_pretrained(output / "final_adapter")
    write_json(output / "train_metrics.json", result.metrics)
    write_json(output / "config.json", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vim27b.yaml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Override the base model ID or local path")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    train(config, args.data, args.output, args.max_steps)


if __name__ == "__main__":
    main()
