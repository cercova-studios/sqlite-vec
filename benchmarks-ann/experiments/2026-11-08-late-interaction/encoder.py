"""CPU-only checkpoint adapter matching inspected ColBERTv2 encoding operations.

Token-marker/masking behavior follows the pinned MIT-licensed ColBERT source;
downloaded source is retained for inspection, never imported or executed.
"""
import json
from pathlib import Path
import string
import time

import numpy as np

HERE = Path(__file__).resolve().parent


class ColBERTEncoder:
    def __init__(self, directory=None):
        import torch
        from safetensors.torch import load_file
        from transformers import BertConfig, BertModel, BertTokenizerFast
        self.torch = torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        directory = Path(directory or HERE / ".work/model")
        metadata = json.loads((directory / "artifact.metadata").read_text())
        assert (metadata["dim"], metadata["query_maxlen"], metadata["doc_maxlen"]) == (128, 32, 180)
        assert metadata["mask_punctuation"] and not metadata["attend_to_mask_tokens"]
        config = BertConfig.from_json_file(str(directory / "config.json"))
        config._attn_implementation = "eager"
        self.model = BertModel(config).eval().to("cpu")
        weights = load_file(str(directory / "model.safetensors"), device="cpu")
        self.projection = weights.pop("linear.weight")
        positions = weights.pop("bert.embeddings.position_ids")
        assert torch.equal(positions, torch.arange(512).reshape(1, 512))
        assert all(k.startswith("bert.") for k in weights)
        self.model.load_state_dict({k[5:]: v for k, v in weights.items()}, strict=True)
        del weights
        self.tokenizer = BertTokenizerFast.from_pretrained(str(directory), local_files_only=True)
        self.skip = {self.tokenizer.encode(c, add_special_tokens=False)[0] for c in string.punctuation}
        self.markers = {"query": self.tokenizer.convert_tokens_to_ids("[unused0]"),
                        "document": self.tokenizer.convert_tokens_to_ids("[unused1]")}
        assert self.markers == {"query": 1, "document": 2}

    def inputs(self, texts, role):
        torch, tok = self.torch, self.tokenizer
        maximum = 32 if role == "query" else 180
        tokens = tok(texts, padding="max_length" if role == "query" else "longest",
                     truncation=True, max_length=maximum - 1, return_tensors="pt")
        ids = torch.cat([tokens["input_ids"][:, :1],
                         torch.full((len(texts), 1), self.markers[role], dtype=torch.long),
                         tokens["input_ids"][:, 1:]], dim=1)
        mask = torch.cat([tokens["attention_mask"][:, :1], torch.ones((len(texts), 1), dtype=torch.long),
                          tokens["attention_mask"][:, 1:]], dim=1)
        if role == "query":
            ids[ids == tok.pad_token_id] = tok.mask_token_id
        return ids, mask

    def encode(self, texts, role):
        if role not in self.markers or not texts or any(not text.strip() for text in texts):
            raise ValueError("nonempty text batch and query/document role required")
        torch = self.torch
        ids, mask = self.inputs(texts, role)
        with torch.inference_mode():
            hidden = self.model(ids, attention_mask=mask, return_dict=False)[0]
            projected = torch.nn.functional.linear(hidden, self.projection)
            output = torch.nn.functional.normalize(projected, p=2, dim=2)
        matrices = []
        for row, token_ids in enumerate(ids.tolist()):
            keep = [n != self.tokenizer.pad_token_id and (role == "query" or n not in self.skip) for n in token_ids]
            matrix = output[row, keep].numpy().astype("<f4")
            assert matrix.shape[1] == 128 and len(matrix) > 0
            assert np.isfinite(matrix).all()
            assert np.all(np.abs(np.einsum("ij,ij->i", matrix.astype(np.float64), matrix.astype(np.float64)) - 1) <= 1e-4)
            matrices.append(matrix)
        return matrices

    def qualification(self):
        examples = ["A short scientific statement.", "Protein expression changes after treatment with the experimental compound."]
        records = {}
        for role in ["query", "document"]:
            ids, mask = self.inputs(examples, role)
            # Independent list construction checks prefix insertion/truncation/padding.
            width = 32 if role == "query" else max(len(self.tokenizer.encode(t)) + 1 for t in examples)
            for text, actual, attention in zip(examples, ids.tolist(), mask.tolist()):
                raw = self.tokenizer.encode(text, max_length=width - 1, truncation=True)
                expected = [raw[0], self.markers[role], *raw[1:]]
                expected_mask = [1] * len(expected) + [0] * (width - len(expected))
                padding = self.tokenizer.mask_token_id if role == "query" else self.tokenizer.pad_token_id
                expected += [padding] * (width - len(expected))
                assert actual == expected and attention == expected_mask
            together = self.encode(examples, role)
            separate = [self.encode([text], role)[0] for text in examples]
            errors = [float(np.max(np.abs(a - b))) for a, b in zip(together, separate)]
            assert max(errors) < 2e-5
            records[role] = dict(token_counts=[len(x) for x in together], padding_max_abs_error=max(errors), marker_ids_valid=True)
        return records


class BGEEncoder:
    def __init__(self, directory):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "model-tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(directory) / "model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])

    def encode(self, texts):
        items = self.tokenizer.encode_batch(["Represent this sentence for searching relevant passages: " + t for t in texts])
        width = max(len(e.ids) for e in items)
        inputs = {name: np.zeros((len(items), width), dtype=np.int64) for name in ["input_ids", "attention_mask", "token_type_ids"]}
        for i, item in enumerate(items):
            size = len(item.ids)
            inputs["input_ids"][i, :size] = item.ids
            inputs["attention_mask"][i, :size] = item.attention_mask
            inputs["token_type_ids"][i, :size] = item.type_ids
        cls = self.session.run(["last_hidden_state"], inputs)[0][:, 0, :]
        return cls / np.sqrt(np.einsum("ij,ij->i", cls, cls))[:, None]
