"""Qwen vision-language model loading and greedy decoding."""

import os


def load_model(source, config, training=False, base_model=None):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=base_model or source,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        max_seq_length=config["max_seq_length"],
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
    )
    if base_model:
        from peft import PeftModel
        from transformers import AutoProcessor

        model = PeftModel.from_pretrained(model, source)
        processor = AutoProcessor.from_pretrained(source)
    if training:
        from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=0,
            bias="none",
            random_state=config["data"]["seed"],
            use_rslora=False,
            loftq_config=None,
        )
        requires_grad_for_gradient_checkpointing(model)
        FastVisionModel.for_training(model)
    else:
        FastVisionModel.for_inference(model)
        processor.tokenizer.padding_side = "left"
    if config["data"].get("single_frame"):
        from transformers.image_utils import SizeDict

        size = processor.video_processor.size
        processor.image_processor.size = SizeDict(
            shortest_edge=int(size["shortest_edge"]) // 4,
            longest_edge=int(size["longest_edge"]) // 4,
        )
    return model, processor


def processor_inputs(processor, examples):
    from unsloth_zoo.vision_utils import collapse_fps, process_vision_info

    conversations = [example["messages"] for example in examples]
    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in conversations
    ]
    kwargs = {"text": texts, "padding": True, "add_special_tokens": False, "return_tensors": "pt"}
    if "images" in examples[0]:
        kwargs["images"] = [image for example in examples for image in example["images"]]
    else:
        images, videos, video_kwargs = process_vision_info(conversations, return_video_kwargs=True)
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
            kwargs["fps"] = collapse_fps(video_kwargs["fps"])
    return processor(**kwargs)


def generate(model, processor, examples, max_new_tokens=80):
    import torch

    inputs = processor_inputs(processor, examples).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    generated = output[:, inputs["input_ids"].shape[-1] :]
    return [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]
