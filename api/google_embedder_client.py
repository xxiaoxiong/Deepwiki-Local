"""Google AI 嵌入模型 ModelClient 集成模块。"""

import os
import logging
import backoff
from typing import Dict, Any, Optional, List, Sequence

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput

try:
    import google.generativeai as genai
    from google.generativeai.types.text_types import EmbeddingDict, BatchEmbeddingDict
except ImportError:
    raise ImportError("需要安装 google-generativeai。请运行 'pip install google-generativeai'")

log = logging.getLogger(__name__)


class GoogleEmbedderClient(ModelClient):
    __doc__ = r"""Google AI 嵌入 API 客户端的组件封装。

    该客户端通过 Google AI API 提供对 Google 嵌入模型的访问。
    支持文本嵌入，适用于语义相似度、检索和分类等任务。

    Args:
        api_key (Optional[str]): Google AI API 密钥，默认为 None。
            未提供时将读取 GOOGLE_API_KEY 环境变量。
        env_api_key_name (str): API 密钥对应的环境变量名，默认为 "GOOGLE_API_KEY"。

    Example:
        ```python
        from api.google_embedder_client import GoogleEmbedderClient
        import adalflow as adal

        client = GoogleEmbedderClient()
        embedder = adal.Embedder(
            model_client=client,
            model_kwargs={
                "model": "gemini-embedding-001",
                "task_type": "SEMANTIC_SIMILARITY"
            }
        )
        ```

    References:
        - Google AI 嵌入文档: https://ai.google.dev/gemini-api/docs/embeddings
        - 可用模型: gemini-embedding-001
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_api_key_name: str = "GOOGLE_API_KEY",
    ):
        """初始化 Google AI 嵌入客户端。

        Args:
            api_key: Google AI API 密钥，未提供时从环境变量读取。
            env_api_key_name: 存储 API 密钥的环境变量名称。
        """
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._initialize_client()  # 初始化 Google AI 客户端

    def _initialize_client(self):
        """使用 API 密钥初始化 Google AI 客户端。"""
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"必须设置环境变量 {self._env_api_key_name}"
            )
        genai.configure(api_key=api_key)  # 配置 Google Generative AI 全局 API 密钥

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """将 Google AI 嵌入响应解析为 EmbedderOutput 格式。

        Args:
            response: Google AI 嵌入响应（EmbeddingDict 或 BatchEmbeddingDict）

        Returns:
            EmbedderOutput: 包含解析后嵌入向量的输出对象
        """
        try:
            from adalflow.core.types import Embedding

            embedding_data = []

            def _extract_embedding_value(obj):
                """从各种格式的响应对象中提取嵌入向量值。"""
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    if "embedding" in obj:
                        return obj.get("embedding")
                    if "embeddings" in obj:
                        return obj.get("embeddings")
                if hasattr(obj, "embedding"):
                    return getattr(obj, "embedding")
                if hasattr(obj, "embeddings"):
                    return getattr(obj, "embeddings")
                # 尝试使用常见序列化方法提取
                for method_name in ("model_dump", "to_dict", "dict"):
                    if hasattr(obj, method_name):
                        try:
                            dumped = getattr(obj, method_name)()
                            if isinstance(dumped, dict):
                                if "embedding" in dumped:
                                    return dumped.get("embedding")
                                if "embeddings" in dumped:
                                    return dumped.get("embeddings")
                        except Exception:
                            pass
                return None

            embedding_value = _extract_embedding_value(response)
            if embedding_value is None:
                log.warning("嵌入响应类型/结构不符合预期: %s", type(response))
                embedding_data = []
            elif isinstance(embedding_value, list) and len(embedding_value) > 0:
                if isinstance(embedding_value[0], (int, float)):
                    # 单个嵌入向量（数字列表）
                    embedding_data = [Embedding(embedding=embedding_value, index=0)]
                elif isinstance(embedding_value[0], list):
                    # 批量嵌入向量（列表的列表）
                    embedding_data = [
                        Embedding(embedding=emb_list, index=i)
                        for i, emb_list in enumerate(embedding_value)
                        if isinstance(emb_list, list) and len(emb_list) > 0
                    ]
                else:
                    # 其他格式，尝试逐项提取
                    extracted = []
                    for item in embedding_value:
                        item_emb = _extract_embedding_value(item)
                        if isinstance(item_emb, list) and len(item_emb) > 0:
                            extracted.append(item_emb)
                    embedding_data = [
                        Embedding(embedding=emb_list, index=i)
                        for i, emb_list in enumerate(extracted)
                    ]
            else:
                log.warning("从响应中解析到的嵌入数据为空或无效")
                embedding_data = []

            if embedding_data:
                first_dim = len(embedding_data[0].embedding) if embedding_data[0].embedding is not None else 0
                log.info("解析了 %s 个嵌入向量（维度=%s）", len(embedding_data), first_dim)

            return EmbedderOutput(
                data=embedding_data,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"解析 Google AI 嵌入响应时出错: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=response
            )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """将输入转换为 Google AI API 所需的参数格式。

        Args:
            input: 要嵌入的文本输入（单个字符串或字符串序列）
            model_kwargs: 模型参数，包括模型名称和 task_type
            model_type: 模型类型，此客户端仅支持 ModelType.EMBEDDER

        Returns:
            Dict: Google AI 嵌入调用所需的 API 参数
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"GoogleEmbedderClient 仅支持 EMBEDDER 类型，当前类型: {model_type}")

        # 确保输入为列表格式
        if isinstance(input, str):
            content = [input]
        elif isinstance(input, Sequence):
            content = list(input)
        else:
            raise TypeError("输入必须是字符串或字符串序列")

        final_model_kwargs = model_kwargs.copy()

        # 根据输入数量选择单个嵌入或批量嵌入参数
        if len(content) == 1:
            final_model_kwargs["content"] = content[0]  # 单个嵌入
        else:
            final_model_kwargs["contents"] = content  # 批量嵌入

        # 未提供 task_type 时使用默认值
        if "task_type" not in final_model_kwargs:
            final_model_kwargs["task_type"] = "SEMANTIC_SIMILARITY"

        # 未提供模型时使用默认模型
        if "model" not in final_model_kwargs:
            final_model_kwargs["model"] = "gemini-embedding-001"

        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Google AI 可能抛出各种异常
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """调用 Google AI 嵌入 API。

        Args:
            api_kwargs: API 调用参数
            model_type: 模型类型，必须为 ModelType.EMBEDDER

        Returns:
            Google AI 嵌入响应对象
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError("GoogleEmbedderClient 仅支持 EMBEDDER 类型")

        # 构造安全日志参数（不记录实际文本内容，避免泄露隐私）
        safe_log_kwargs = {k: v for k, v in api_kwargs.items() if k not in {"content", "contents"}}
        if "content" in api_kwargs:
            safe_log_kwargs["content_chars"] = len(str(api_kwargs.get("content", "")))
        if "contents" in api_kwargs:
            try:
                contents = api_kwargs.get("contents")
                safe_log_kwargs["contents_count"] = len(contents) if hasattr(contents, "__len__") else None
            except Exception:
                safe_log_kwargs["contents_count"] = None
        log.info("Google AI 嵌入调用参数（已脱敏）: %s", safe_log_kwargs)

        try:
            if "content" in api_kwargs:
                # 单个文本嵌入
                response = genai.embed_content(**api_kwargs)
            elif "contents" in api_kwargs:
                # 批量文本嵌入（Google AI 原生支持批量）
                # 复制一份避免修改原始参数（重试时需要原始参数）
                kwargs = api_kwargs.copy()
                contents = kwargs.pop("contents")
                response = genai.embed_content(content=contents, **kwargs)
            else:
                raise ValueError("必须提供 'content' 或 'contents' 参数")

            return response

        except Exception as e:
            log.error(f"调用 Google AI 嵌入 API 时出错: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """异步调用 Google AI 嵌入 API。

        注意：Google AI Python 客户端暂不支持原生异步，
        此方法将回退到同步调用。
        """
        # Google AI 客户端暂无原生异步支持，直接调用同步方法
        return self.call(api_kwargs, model_type)
