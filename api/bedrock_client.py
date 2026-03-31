"""AWS Bedrock ModelClient 集成模块。"""

import os
import json
import logging
import boto3
import botocore
import backoff
from typing import Dict, Any, Optional, List, Generator, Union, AsyncGenerator, Sequence

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput, EmbedderOutput

# 配置日志
from api.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)


class BedrockClient(ModelClient):
    __doc__ = r"""AWS Bedrock API 客户端的组件封装。

    AWS Bedrock 提供统一的 API，可访问多种基础模型，
    包括 Amazon 自研模型和第三方模型（如 Anthropic Claude）。

    Example:
        ```python
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        )
        ```
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_role_arn: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        """初始化 AWS Bedrock 客户端。

        Args:
            aws_access_key_id: AWS 访问密钥 ID，未提供时读取环境变量 AWS_ACCESS_KEY_ID。
            aws_secret_access_key: AWS 秘密访问密钥，未提供时读取环境变量 AWS_SECRET_ACCESS_KEY。
            aws_session_token: AWS 会话令牌，未提供时读取环境变量 AWS_SESSION_TOKEN。
            aws_region: AWS 区域，未提供时读取环境变量 AWS_REGION。
            aws_role_arn: AWS IAM 角色 ARN（角色扮演认证），未提供时读取环境变量 AWS_ROLE_ARN。
        """
        super().__init__(*args, **kwargs)
        from api.config import (
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
            AWS_SESSION_TOKEN,
            AWS_REGION,
            AWS_ROLE_ARN,
        )

        # 优先使用传入参数，其次读取配置/环境变量
        self.aws_access_key_id = aws_access_key_id or AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = aws_secret_access_key or AWS_SECRET_ACCESS_KEY
        self.aws_session_token = aws_session_token or AWS_SESSION_TOKEN
        self.aws_region = aws_region or AWS_REGION or "us-east-1"  # 默认区域
        self.aws_role_arn = aws_role_arn or AWS_ROLE_ARN

        self.sync_client = self.init_sync_client()  # 初始化同步客户端
        self.async_client = None  # 异步客户端按需初始化

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建实例。"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """将实例序列化为字典。"""
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
            "aws_region": self.aws_region,
            "aws_role_arn": self.aws_role_arn,
        }

    def __getstate__(self):
        """自定义序列化逻辑，排除不可 pickle 的客户端对象。
        由 pickle 保存对象状态时调用。
        """
        state = self.__dict__.copy()
        # 移除不可序列化的客户端实例
        if 'sync_client' in state:
            del state['sync_client']
        if 'async_client' in state:
            del state['async_client']
        return state

    def __setstate__(self, state):
        """自定义反序列化逻辑，重新创建客户端对象。
        由 pickle 恢复对象状态时调用。
        """
        self.__dict__.update(state)
        # 反序列化后重新初始化客户端
        self.sync_client = self.init_sync_client()
        self.async_client = None  # 异步客户端按需懒加载

    def init_sync_client(self):
        """初始化同步 AWS Bedrock 客户端。"""
        try:
            # 使用提供的凭证创建 boto3 会话
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region
            )

            # 若提供了角色 ARN，则扮演该角色
            if self.aws_role_arn:
                sts_client = session.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.aws_role_arn,
                    RoleSessionName="DeepWikiBedrockSession"
                )
                credentials = assumed_role['Credentials']
                # 使用扮演角色的临时凭证创建新会话
                session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.aws_region
                )

            # 创建 Bedrock Runtime 客户端
            bedrock_runtime = session.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region
            )
            return bedrock_runtime

        except Exception as e:
            log.error(f"初始化 AWS Bedrock 客户端时出错: {str(e)}")
            return None  # 返回 None 表示初始化失败

    def init_async_client(self):
        """初始化异步 AWS Bedrock 客户端。

        注意：boto3 不原生支持异步，此处复用同步客户端，
        异步行为在上层处理。
        """
        return self.sync_client

    def _get_model_provider(self, model_id: str) -> str:
        """从模型 ID 中提取提供商名称。

        Args:
            model_id: 模型推理 ID，例如：
                - "anthropic.claude-3-sonnet-20240229-v1:0"
                - "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
                - "global.cohere.embed-v4:0"

        Returns:
            提供商名称，例如 "anthropic"
        """
        seg = model_id.split(".")
        if len(seg) >= 3:
            return seg[1]  # 跨区域格式：global.provider.model
        elif len(seg) == 2:
            return seg[0]  # 标准格式：provider.model
        else:
            return "amazon"  # 无法解析时默认为 Amazon

    def _format_prompt_for_provider(self, provider: str, prompt: str, messages=None) -> Dict[str, Any]:
        """根据提供商要求格式化请求体。

        Args:
            provider: 提供商名称，例如 "anthropic"
            prompt: 输入文本
            messages: 可选的对话消息列表（用于聊天模型）

        Returns:
            格式化后的请求体字典
        """
        if provider == "anthropic":
            # Claude 模型格式
            if messages:
                formatted_messages = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    formatted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": msg.get("content", "")}]
                    })
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": formatted_messages,
                    "max_tokens": 4096
                }
            else:
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    "max_tokens": 4096
                }
        elif provider == "amazon":
            # Amazon Titan 模型格式
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.8
                }
            }
        elif provider == "cohere":
            # Cohere 模型格式
            return {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "p": 0.8
            }
        elif provider == "ai21":
            # AI21 模型格式
            return {
                "prompt": prompt,
                "maxTokens": 4096,
                "temperature": 0.7,
                "topP": 0.8
            }
        else:
            # 默认格式（兜底）
            return {"prompt": prompt}

    def _extract_response_text(self, provider: str, response: Dict[str, Any]) -> str:
        """从 Bedrock API 响应中提取生成文本。

        Args:
            provider: 提供商名称，例如 "anthropic"
            response: Bedrock API 返回的响应字典

        Returns:
            生成的文本字符串
        """
        if provider == "anthropic":
            return response.get("content", [{}])[0].get("text", "")
        elif provider == "amazon":
            return response.get("results", [{}])[0].get("outputText", "")
        elif provider == "cohere":
            return response.get("generations", [{}])[0].get("text", "")
        elif provider == "ai21":
            return response.get("completions", [{}])[0].get("data", {}).get("text", "")
        else:
            # 尝试从常见字段提取文本
            if isinstance(response, dict):
                for key in ["text", "content", "output", "completion"]:
                    if key in response:
                        return response[key]
            return str(response)

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """将 Bedrock 嵌入响应解析为 EmbedderOutput 格式。"""
        from adalflow.core.types import Embedding

        try:
            embedding_data: List[Embedding] = []

            if isinstance(response, dict) and "embeddings" in response:
                # 批量嵌入响应
                embeddings = response.get("embeddings") or []
                embedding_data = [
                    Embedding(embedding=emb, index=i) for i, emb in enumerate(embeddings)
                ]
            elif isinstance(response, dict) and "embedding" in response:
                # 单个嵌入响应
                emb = response.get("embedding") or []
                embedding_data = [Embedding(embedding=emb, index=0)]
            else:
                raise ValueError(f"不支持的嵌入响应类型: {type(response)}")

            return EmbedderOutput(data=embedding_data, error=None, raw_response=response)
        except Exception as e:
            log.error(f"解析 Bedrock 嵌入响应时出错: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    @backoff.on_exception(
        backoff.expo,
        (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """同步调用 AWS Bedrock API。"""
        api_kwargs = api_kwargs or {}

        # 检查客户端是否已正确初始化
        if not self.sync_client:
            error_msg = "AWS Bedrock 客户端未初始化，请检查 AWS 凭证和区域设置。"
            log.error(error_msg)
            return error_msg

        if model_type == ModelType.LLM:
            model_id = api_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            provider = self._get_model_provider(model_id)

            prompt = api_kwargs.get("input", "")
            messages = api_kwargs.get("messages")

            # 按提供商格式化请求体
            request_body = self._format_prompt_for_provider(provider, prompt, messages)

            # 注入 temperature 参数（各提供商字段名不同）
            if "temperature" in api_kwargs:
                if provider == "anthropic":
                    request_body["temperature"] = api_kwargs["temperature"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["temperature"] = api_kwargs["temperature"]
                elif provider in ("cohere", "ai21"):
                    request_body["temperature"] = api_kwargs["temperature"]

            # 注入 top_p 参数
            if "top_p" in api_kwargs:
                if provider == "anthropic":
                    request_body["top_p"] = api_kwargs["top_p"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["topP"] = api_kwargs["top_p"]
                elif provider == "cohere":
                    request_body["p"] = api_kwargs["top_p"]
                elif provider == "ai21":
                    request_body["topP"] = api_kwargs["top_p"]

            body = json.dumps(request_body)

            try:
                response = self.sync_client.invoke_model(
                    modelId=model_id,
                    body=body
                )
                response_body = json.loads(response["body"].read())
                return self._extract_response_text(provider, response_body)

            except Exception as e:
                log.error(f"调用 AWS Bedrock API 时出错: {str(e)}")
                return f"错误: {str(e)}"

        elif model_type == ModelType.EMBEDDER:
            model_id = api_kwargs.get("model", "amazon.titan-embed-text-v2:0")
            provider = self._get_model_provider(model_id)
            texts = api_kwargs.get("input", [])
            model_kwargs = api_kwargs.get("model_kwargs") or {}
            embeddings: List[List[float]] = []
            raw_responses: List[Dict[str, Any]] = []

            if provider == "amazon":
                # Amazon Titan Embed 不支持批量，逐条发送
                for text in texts:
                    request_body: Dict[str, Any] = {"inputText": text}
                    dimensions = model_kwargs.get("dimensions")
                    if dimensions is not None:
                        request_body["dimensions"] = int(dimensions)
                    normalize = model_kwargs.get("normalize")
                    if normalize is not None:
                        request_body["normalize"] = bool(normalize)

                    response = self.sync_client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(request_body),
                    )
                    response_body = json.loads(response["body"].read())
                    raw_responses.append(response_body)
                    emb = response_body.get("embedding")
                    if emb is None:
                        raise ValueError(f"响应中未找到嵌入向量: {response_body}")
                    embeddings.append(emb)

            elif provider == "cohere":
                # Cohere 支持批量，一次发送所有文本
                request_body = {
                    "texts": texts,
                    "input_type": model_kwargs.get("input_type") or "search_document",
                }
                response = self.sync_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                )
                response_body = json.loads(response["body"].read())
                raw_responses.append(response_body)
                batch_embeddings = response_body.get("embeddings")
                if isinstance(batch_embeddings, list):
                    embeddings = batch_embeddings
                elif isinstance(batch_embeddings, dict) and "float" in batch_embeddings:
                    embeddings = batch_embeddings["float"]
                else:
                    raise ValueError(f"响应中未找到嵌入向量: {response_body}")
            else:
                raise NotImplementedError(f"Bedrock 客户端不支持嵌入提供商 '{provider}'")

            return {"embeddings": embeddings, "raw_responses": raw_responses}
        else:
            raise ValueError(f"AWS Bedrock 客户端不支持模型类型 {model_type}")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """异步调用 AWS Bedrock API。

        注意：当前实现直接调用同步方法。
        生产环境建议使用线程池（asyncio.to_thread）执行同步调用。
        """
        return self.call(api_kwargs, model_type)

    def convert_inputs_to_api_kwargs(
        self, input: Any = None, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """将 AdalFlow 标准输入转换为 AWS Bedrock API 所需格式。"""
        model_kwargs = model_kwargs or {}
        api_kwargs = {}

        if model_type == ModelType.LLM:
            api_kwargs["model"] = model_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            api_kwargs["input"] = input
            if "temperature" in model_kwargs:
                api_kwargs["temperature"] = model_kwargs["temperature"]
            if "top_p" in model_kwargs:
                api_kwargs["top_p"] = model_kwargs["top_p"]
            return api_kwargs

        elif model_type == ModelType.EMBEDDER:
            # 统一输入格式为字符串列表
            if isinstance(input, str):
                inputs = [input]
            elif isinstance(input, Sequence):
                inputs = list(input)
            else:
                raise TypeError("输入必须是字符串或字符串序列")

            api_kwargs["model"] = model_kwargs.get("model", "amazon.titan-embed-text-v2:0")
            api_kwargs["input"] = inputs
            api_kwargs["model_kwargs"] = model_kwargs
            return api_kwargs
        else:
            raise ValueError(f"AWS Bedrock 客户端不支持模型类型 {model_type}")


            