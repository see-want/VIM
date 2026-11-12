"""MPNet and SigLIP text embeddings used by the paper metrics."""

import numpy as np


def encode(texts, source, kind, device="cuda", batch_size=32):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, SiglipModel

    torch.backends.cuda.matmul.allow_tf32 = False
    if kind == "sbert":
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModel.from_pretrained(source).to(device).eval()
        max_length = 384
    else:
        tokenizer = AutoProcessor.from_pretrained(source).tokenizer
        model = (
            SiglipModel.from_pretrained(
                source, torch_dtype=torch.float32, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        max_length = 64
    unique = sorted({text for text in texts if text.strip()})
    vectors = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length" if kind == "siglip" else True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            if kind == "sbert":
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                features = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            else:
                features = model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
            features = F.normalize(features.float(), dim=-1).cpu().numpy()
        vectors.update(zip(batch, features))
    vectors[""] = np.zeros_like(next(iter(vectors.values())))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return vectors
