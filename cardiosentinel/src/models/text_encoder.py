"""
Stream 2 (clinical notes) and Stream 3 (symptom descriptions) encoders: thin
wrappers around a pretrained HuggingFace encoder (ClinicalBERT / BioBERT) that
return the pooled [CLS] embedding. `freeze` controls whether the backbone is
fine-tuned (GPU config) or used as a fixed feature extractor (CPU/smoke config,
where fine-tuning a 110M-parameter model is not practical) — see configs/.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel

CLINICALBERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BIOBERT_NAME = "dmis-lab/biobert-base-cased-v1.1"
# dmis-lab/biobert-base-cased-v1.1 ships an incomplete upload: config.json
# has no tokenizer files at all, AND no `model_type` key, so AutoTokenizer/
# AutoModel can't auto-detect anything. BioBERT v1.1 was continued-pretrained
# from bert-base-cased WITHOUT changing the vocabulary or architecture
# (confirmed: vocab_size=28996/hidden=768/12 layers, identical to
# bert-base-cased) — so sourcing the tokenizer from bert-base-cased and
# loading weights via the concrete BertModel class (bypassing AutoModel's
# model_type lookup) is correct, not just a workaround.
BIOBERT_TOKENIZER_NAME = "bert-base-cased"


class PretrainedTextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False, max_length: int = 256,
                 tokenizer_name: str = None, model_cls=AutoModel):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.backbone = model_cls.from_pretrained(model_name)
        self.max_length = max_length
        self.output_dim = self.backbone.config.hidden_size
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [CLS]


def build_clinicalbert(freeze: bool = False, max_length: int = 256) -> PretrainedTextEncoder:
    return PretrainedTextEncoder(CLINICALBERT_NAME, freeze=freeze, max_length=max_length)


def build_biobert(freeze: bool = False, max_length: int = 128) -> PretrainedTextEncoder:
    return PretrainedTextEncoder(BIOBERT_NAME, freeze=freeze, max_length=max_length,
                                  tokenizer_name=BIOBERT_TOKENIZER_NAME, model_cls=BertModel)
