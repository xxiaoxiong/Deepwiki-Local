"""
本地 Sentence Transformers 嵌入器，兼容 adalflow FAISSRetriever 接口。
支持 BAAI/bge-small-zh-v1.5 等 sentence-transformers 模型。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import logging
from copy import deepcopy

from adalflow.core.types import Document
from adalflow.core.component import DataComponent

logger = logging.getLogger(__name__)


@dataclass
class STEmbeddingItem:
    """单条嵌入结果，兼容 adalflow EmbedderOutput.data 的访问方式。"""
    embedding: List[float]
    index: int = 0


@dataclass
class STEmbedderOutput:
    """嵌入输出，兼容 adalflow EmbedderOutput 接口（`.data[0].embedding`）。"""
    data: List[STEmbeddingItem] = field(default_factory=list)
    error: Optional[str] = None


class SentenceTransformerEmbedder:
    """
    基于 sentence-transformers 的本地嵌入器。
    兼容 adalflow 的 FAISSRetriever（query 嵌入）接口。
    调用方式：embedder(input=text_or_list) → STEmbedderOutput
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            path = self._model_path or self._model_name
            logger.info(f"加载 sentence-transformer 模型: {path}")
            self._model = SentenceTransformer(path, device=self._device)
            logger.info(f"模型加载成功，嵌入维度: {self._model.get_sentence_embedding_dimension()}")

    def __call__(self, input, **kwargs):  # noqa: A002
        """
        嵌入一条或多条文本。
        Args:
            input: 单个字符串或字符串列表
        Returns:
            STEmbedderOutput，与 adalflow EmbedderOutput 格式兼容
        """
        self._ensure_loaded()
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        embeddings = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        data = [STEmbeddingItem(embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]
        return STEmbedderOutput(data=data)


class SentenceTransformerDocumentProcessor(DataComponent):
    """
    基于 sentence-transformers 的批量文档嵌入组件。
    与 OllamaDocumentProcessor 接口一致，用于文档索引阶段。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_path
        self._device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            path = self._model_path or self._model_name
            logger.info(f"加载 sentence-transformer 模型: {path}")
            self._model = SentenceTransformer(path, device=self._device)

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        """
        批量处理文档，为每个文档生成嵌入向量并设置 doc.vector。

        Args:
            documents: 待处理的文档序列

        Returns:
            包含有效嵌入向量的文档列表
        """
        self._ensure_loaded()
        output = deepcopy(documents)
        logger.info(f"使用 sentence-transformers 嵌入 {len(output)} 个文档")

        texts = [doc.text for doc in output]
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            valid_docs = []
            expected_dim: Optional[int] = None

            for i, (doc, emb) in enumerate(zip(output, embeddings)):
                emb_list = emb.tolist()

                if expected_dim is None:
                    expected_dim = len(emb_list)
                    logger.info(f"嵌入向量维度: {expected_dim}")
                elif len(emb_list) != expected_dim:
                    file_path = getattr(doc, "meta_data", {}).get("file_path", f"document_{i}")
                    logger.warning(
                        f"跳过文档 '{file_path}'：维度不一致 {len(emb_list)} != {expected_dim}"
                    )
                    continue

                output[i].vector = emb_list
                valid_docs.append(output[i])

            logger.info(f"成功嵌入 {len(valid_docs)}/{len(output)} 个文档")
            return valid_docs

        except Exception as e:
            logger.error(f"文档嵌入失败: {e}")
            raise
