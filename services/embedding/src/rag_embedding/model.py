"""Sentence Transformers adapter, imported only by the separate runtime."""

from typing import Protocol

from rag_embedding.config import Settings


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]: ...


class QwenEncoder:
    def __init__(self, settings: Settings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(settings.embedding_threads)
        self.model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            cache_folder=str(settings.embedding_cache_path),
            device="cpu",
            trust_remote_code=False,
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = settings.embedding_max_tokens
        if self.model.get_embedding_dimension() != 1024:
            raise ValueError("Unexpected model dimensions")
        self.batch_size = settings.embedding_batch_size

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        tokens = self.model.preprocess(texts)
        count = int(tokens["attention_mask"].sum().item())
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            prompt="",
        )
        return vectors.tolist(), count
