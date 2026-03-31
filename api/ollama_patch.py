from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import requests
import os

# 配置日志
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class OllamaModelNotFoundError(Exception):
    """当 Ollama 模型未找到时抛出的自定义异常。"""
    pass


def check_ollama_model_exists(model_name: str, ollama_host: str = None) -> bool:
    """
    在尝试使用 Ollama 模型之前，检查该模型是否存在。

    Args:
        model_name: 要检查的模型名称
        ollama_host: Ollama 服务地址，默认使用 localhost:11434

    Returns:
        bool: 模型存在返回 True，否则返回 False
    """
    if ollama_host is None:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        # 去除末尾的 /api 前缀（若有），然后重新拼接
        if ollama_host.endswith('/api'):
            ollama_host = ollama_host[:-4]

        # 请求 Ollama 已安装模型列表
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            # 获取所有可用模型的基础名称（去掉 tag 部分）
            available_models = [model.get('name', '').split(':')[0] for model in models_data.get('models', [])]
            model_base_name = model_name.split(':')[0]  # 去掉模型 tag

            is_available = model_base_name in available_models
            if is_available:
                logger.info(f"Ollama 模型 '{model_name}' 可用")
            else:
                logger.warning(f"Ollama 模型 '{model_name}' 不可用。可用模型: {available_models}")
            return is_available
        else:
            logger.warning(f"无法检查 Ollama 模型，状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"无法连接到 Ollama 以检查模型: {e}")
        return False
    except Exception as e:
        logger.warning(f"检查 Ollama 模型可用性时出错: {e}")
        return False


class OllamaDocumentProcessor(DataComponent):
    """
    为 Ollama 嵌入模型逐个处理文档的数据组件。
    由于 Adalflow 的 Ollama 客户端不支持批量嵌入，
    因此需要对每个文档单独进行处理。
    """

    def __init__(self, embedder: adal.Embedder) -> None:
        super().__init__()
        self.embedder = embedder  # 用于生成文档嵌入向量的嵌入器

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        """
        对文档列表逐个生成 Ollama 嵌入向量。

        Args:
            documents: 待处理的文档序列

        Returns:
            包含有效嵌入向量的文档列表（大小不一致的文档会被跳过）
        """
        output = deepcopy(documents)  # 深拷贝，避免修改原始数据
        logger.info(f"正在逐个处理 {len(output)} 个文档以生成 Ollama 嵌入向量")

        successful_docs = []  # 成功处理的文档列表
        expected_embedding_size = None  # 期望的嵌入向量维度（由第一个成功文档确定）

        for i, doc in enumerate(tqdm(output, desc="正在为 Ollama 嵌入处理文档")):
            try:
                # 为单个文档生成嵌入向量
                result = self.embedder(input=doc.text)
                if result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # 验证嵌入向量维度一致性
                    if expected_embedding_size is None:
                        # 以第一个成功文档的维度为基准
                        expected_embedding_size = len(embedding)
                        logger.info(f"期望嵌入向量维度设为: {expected_embedding_size}")
                    elif len(embedding) != expected_embedding_size:
                        # 维度不一致的文档跳过，避免 FAISS 报错
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"文档 '{file_path}' 的嵌入维度不一致 {len(embedding)} != {expected_embedding_size}，已跳过")
                        continue

                    # 将嵌入向量赋值给文档
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"无法为文档 '{file_path}' 生成嵌入向量，已跳过")
            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error(f"处理文档 '{file_path}' 时出错: {e}，已跳过")

        logger.info(f"成功处理 {len(successful_docs)}/{len(output)} 个文档，嵌入维度一致")
        return successful_docs
