"""OpenRouter ModelClient 集成模块。"""

from typing import Dict, Sequence, Optional, Any, List
import logging
import json
import aiohttp
import requests
from requests.exceptions import RequestException, Timeout

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    CompletionUsage,
    ModelType,
    GeneratorOutput,
)

log = logging.getLogger(__name__)


class OpenRouterClient(ModelClient):
    __doc__ = r"""OpenRouter API 客户端的组件封装。

    OpenRouter 提供统一的 API，通过单个端点访问数百个 AI 模型。
    其 API 格式与 OpenAI 兼容，仅有少量差异。

    详情请访问 https://openrouter.ai/docs

    Example:
        ```python
        from api.openrouter_client import OpenRouterClient

        client = OpenRouterClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "openai/gpt-4o"}
        )
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        """初始化 OpenRouter 客户端。"""
        super().__init__(*args, **kwargs)
        self.sync_client = self.init_sync_client()  # 初始化同步客户端
        self.async_client = None  # 异步客户端懒加载，仅在需要时初始化

    def init_sync_client(self):
        """初始化同步 OpenRouter 客户端（使用 requests 库）。"""
        from api.config import OPENROUTER_API_KEY
        api_key = OPENROUTER_API_KEY
        if not api_key:
            log.warning("OPENROUTER_API_KEY 未配置")
        # OpenRouter 没有专属客户端库，直接使用 requests
        return {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1"
        }

    def init_async_client(self):
        """初始化异步 OpenRouter 客户端（使用 aiohttp 库）。"""
        from api.config import OPENROUTER_API_KEY
        api_key = OPENROUTER_API_KEY
        if not api_key:
            log.warning("OPENROUTER_API_KEY 未配置")
        # 异步请求使用 aiohttp
        return {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1"
        }

    def convert_inputs_to_api_kwargs(
        self, input: Any, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """将 AdalFlow 标准输入转换为 OpenRouter API 所需格式。"""
        model_kwargs = model_kwargs or {}

        if model_type == ModelType.LLM:
            messages = []
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list) and all(isinstance(msg, dict) for msg in input):
                messages = input
            else:
                raise ValueError(f"OpenRouter 不支持的输入格式: {type(input)}")

            log.info(f"发送给 OpenRouter 的消息: {messages}")
            api_kwargs = {"messages": messages, **model_kwargs}
            # 未指定模型时使用默认模型
            if "model" not in api_kwargs:
                api_kwargs["model"] = "openai/gpt-3.5-turbo"
            return api_kwargs

        elif model_type == ModelType.EMBEDDING:
            # OpenRouter 暂不直接支持嵌入接口
            raise NotImplementedError("OpenRouter 客户端暂不支持嵌入功能")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """异步调用 OpenRouter API。"""
        if not self.async_client:
            self.async_client = self.init_async_client()

        # 检查 API Key 是否已配置
        if not self.async_client.get("api_key"):
            error_msg = "OPENROUTER_API_KEY 未配置。请设置该环境变量以使用 OpenRouter。"
            log.error(error_msg)
            async def error_generator():
                yield error_msg
            return error_generator()

        api_kwargs = api_kwargs or {}

        if model_type == ModelType.LLM:
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.async_client['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AsyncFuncAI/deepwiki-open",
                "X-Title": "DeepWiki"
            }
            # OpenRouter 统一使用非流式模式
            api_kwargs["stream"] = False

            try:
                log.info(f"正在向 OpenRouter 发起异步请求: {self.async_client['base_url']}/chat/completions")

                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(
                            f"{self.async_client['base_url']}/chat/completions",
                            headers=headers,
                            json=api_kwargs,
                            timeout=60
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                log.error(f"OpenRouter API 错误 ({response.status}): {error_text}")
                                async def error_response_generator():
                                    yield f"OpenRouter API 错误 ({response.status}): {error_text}"
                                return error_response_generator()

                            # 获取完整响应
                            data = await response.json()
                            log.info(f"收到 OpenRouter 响应: {data}")

                            async def content_generator():
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "message" in choice and "content" in choice["message"]:
                                        content = choice["message"]["content"]
                                        log.info("成功获取响应内容")

                                        # 如果内容是 XML 格式，进行特殊处理
                                        if content.strip().startswith("<") and ">" in content:
                                            try:
                                                xml_content = content
                                                # 专门处理 wiki_structure XML
                                                if "<wiki_structure>" in xml_content:
                                                    log.info("检测到 wiki_structure XML，确保格式正确")
                                                    import re
                                                    wiki_match = re.search(r'<wiki_structure>[\s\S]*?<\/wiki_structure>', xml_content)
                                                    if wiki_match:
                                                        raw_xml = wiki_match.group(0)
                                                        clean_xml = raw_xml.strip()
                                                        try:
                                                            # 修复常见 XML 问题
                                                            fixed_xml = clean_xml
                                                            # 将裸 & 替换为 &amp;（已是实体引用的不替换）
                                                            fixed_xml = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', fixed_xml)
                                                            fixed_xml = fixed_xml.replace('</', '</').replace('  >', '>')
                                                            from xml.dom.minidom import parseString
                                                            dom = parseString(fixed_xml)
                                                            pretty_xml = dom.toprettyxml()
                                                            # 去掉 XML 声明头
                                                            if pretty_xml.startswith('<?xml'):
                                                                pretty_xml = pretty_xml[pretty_xml.find('?>')+2:].strip()
                                                            log.info(f"提取并验证 XML 成功: {pretty_xml[:100]}...")
                                                            yield pretty_xml
                                                        except Exception as xml_parse_error:
                                                            log.warning(f"XML 验证失败: {str(xml_parse_error)}，使用原始 XML")
                                                            # XML 验证失败，尝试重建结构
                                                            try:
                                                                structure_match = re.search(r'<wiki_structure>(.*?)</wiki_structure>', clean_xml, re.DOTALL)
                                                                if structure_match:
                                                                    structure = structure_match.group(1).strip()
                                                                    clean_structure = "<wiki_structure>\n"
                                                                    title_match = re.search(r'<title>(.*?)</title>', structure, re.DOTALL)
                                                                    if title_match:
                                                                        clean_structure += f"  <title>{title_match.group(1).strip()}</title>\n"
                                                                    desc_match = re.search(r'<description>(.*?)</description>', structure, re.DOTALL)
                                                                    if desc_match:
                                                                        clean_structure += f"  <description>{desc_match.group(1).strip()}</description>\n"
                                                                    clean_structure += "  <pages>\n"
                                                                    pages = re.findall(r'<page id="(.*?)">(.*?)</page>', structure, re.DOTALL)
                                                                    for page_id, page_content in pages:
                                                                        clean_structure += f'    <page id="{page_id}">\n'
                                                                        page_title_match = re.search(r'<title>(.*?)</title>', page_content, re.DOTALL)
                                                                        if page_title_match:
                                                                            clean_structure += f"      <title>{page_title_match.group(1).strip()}</title>\n"
                                                                        page_desc_match = re.search(r'<description>(.*?)</description>', page_content, re.DOTALL)
                                                                        if page_desc_match:
                                                                            clean_structure += f"      <description>{page_desc_match.group(1).strip()}</description>\n"
                                                                        importance_match = re.search(r'<importance>(.*?)</importance>', page_content, re.DOTALL)
                                                                        if importance_match:
                                                                            clean_structure += f"      <importance>{importance_match.group(1).strip()}</importance>\n"
                                                                        clean_structure += "      <relevant_files>\n"
                                                                        for fp in re.findall(r'<file_path>(.*?)</file_path>', page_content, re.DOTALL):
                                                                            clean_structure += f"        <file_path>{fp.strip()}</file_path>\n"
                                                                        clean_structure += "      </relevant_files>\n"
                                                                        clean_structure += "      <related_pages>\n"
                                                                        for related in re.findall(r'<related>(.*?)</related>', page_content, re.DOTALL):
                                                                            clean_structure += f"        <related>{related.strip()}</related>\n"
                                                                        clean_structure += "      </related_pages>\n"
                                                                        clean_structure += "    </page>\n"
                                                                    clean_structure += "  </pages>\n</wiki_structure>"
                                                                    log.info("成功重建干净的 XML 结构")
                                                                    yield clean_structure
                                                                else:
                                                                    log.warning("无法提取 wiki 结构，使用原始 XML")
                                                                    yield clean_xml
                                                            except Exception as rebuild_error:
                                                                log.warning(f"XML 重建失败: {str(rebuild_error)}，使用原始 XML")
                                                                yield clean_xml
                                                    else:
                                                        log.warning("无法提取 wiki_structure XML，返回原始内容")
                                                        yield xml_content
                                                else:
                                                    # 其他 XML 内容直接返回
                                                    yield content
                                            except Exception as xml_error:
                                                log.error(f"处理 XML 内容时出错: {str(xml_error)}")
                                                yield content
                                        else:
                                            # 非 XML 内容直接返回
                                            yield content
                                    else:
                                        log.error(f"响应格式异常: {data}")
                                        yield "错误: OpenRouter API 响应格式异常"
                                else:
                                    log.error(f"响应中无 choices 字段: {data}")
                                    yield "错误: OpenRouter API 无响应内容"

                            return content_generator()
                    except aiohttp.ClientError as e:
                        e_client = e
                        log.error(f"与 OpenRouter API 连接出错: {str(e_client)}")
                        async def connection_error_generator():
                            yield f"与 OpenRouter API 连接出错: {str(e_client)}。请检查网络连接及 OpenRouter API 是否可访问。"
                        return connection_error_generator()

            except RequestException as e:
                e_req = e
                log.error(f"异步调用 OpenRouter API 时出错: {str(e_req)}")
                async def request_error_generator():
                    yield f"调用 OpenRouter API 时出错: {str(e_req)}"
                return request_error_generator()

            except Exception as e:
                e_unexp = e
                log.error(f"异步调用 OpenRouter API 时发生未知错误: {str(e_unexp)}")
                async def unexpected_error_generator():
                    yield f"调用 OpenRouter API 时发生未知错误: {str(e_unexp)}"
                return unexpected_error_generator()

        else:
            error_msg = f"不支持的模型类型: {model_type}"
            log.error(error_msg)
            async def model_type_error_generator():
                yield error_msg
            return model_type_error_generator()

    def _process_completion_response(self, data: Dict) -> GeneratorOutput:
        """处理 OpenRouter 的非流式补全响应。"""
        try:
            if not data.get("choices"):
                raise ValueError(f"OpenRouter 响应中无 choices 字段: {data}")

            choice = data["choices"][0]
            # 提取补全文本
            if "message" in choice:
                content = choice["message"].get("content", "")
            elif "text" in choice:
                content = choice.get("text", "")
            else:
                raise ValueError(f"OpenRouter 响应格式异常: {choice}")

            # 提取 token 用量信息（如有）
            usage = None
            if "usage" in data:
                usage = CompletionUsage(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0)
                )

            return GeneratorOutput(data=content, usage=usage, raw_response=data)

        except Exception as e_proc:
            log.error(f"处理 OpenRouter 补全响应时出错: {str(e_proc)}")
            raise

    def _process_streaming_response(self, response):
        """处理 OpenRouter 的同步流式响应（SSE 格式）。"""
        try:
            log.info("开始处理 OpenRouter 流式响应")
            buffer = ""

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                try:
                    buffer += chunk
                    # 逐行处理缓冲区中的完整行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        # 跳过 SSE 注释行（以 : 开头）
                        if line.startswith(':'):
                            continue
                        if line.startswith("data: "):
                            data = line[6:]  # 去掉 "data: " 前缀
                            if data == "[DONE]":
                                log.info("收到 [DONE] 标志，流式响应结束")
                                break
                            try:
                                data_obj = json.loads(data)
                                if "choices" in data_obj and len(data_obj["choices"]) > 0:
                                    choice = data_obj["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                        yield choice["delta"]["content"]
                                    elif "text" in choice:
                                        yield choice["text"]
                            except json.JSONDecodeError:
                                log.warning(f"无法解析 SSE 数据: {data}")
                                continue
                except Exception as e_chunk:
                    log.error(f"处理流式数据块时出错: {str(e_chunk)}")
                    yield f"处理响应数据块时出错: {str(e_chunk)}"
        except Exception as e_stream:
            log.error(f"流式响应处理出错: {str(e_stream)}")
            yield f"流式响应处理出错: {str(e_stream)}"

    async def _process_async_streaming_response(self, response):
        """处理 OpenRouter 的异步流式响应（SSE 格式）。"""
        buffer = ""
        try:
            log.info("开始处理 OpenRouter 异步流式响应")
            async for chunk in response.content:
                try:
                    # 将字节转为字符串并追加到缓冲区
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode('utf-8')
                    else:
                        chunk_str = str(chunk)

                    buffer += chunk_str

                    # 逐行处理缓冲区中的完整行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        # 跳过 SSE 注释行
                        if line.startswith(':'):
                            continue
                        if line.startswith("data: "):
                            data = line[6:]  # 去掉 "data: " 前缀
                            if data == "[DONE]":
                                log.info("收到 [DONE] 标志，流式响应结束")
                                break
                            try:
                                data_obj = json.loads(data)
                                if "choices" in data_obj and len(data_obj["choices"]) > 0:
                                    choice = data_obj["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                        yield choice["delta"]["content"]
                                    elif "text" in choice:
                                        yield choice["text"]
                            except json.JSONDecodeError:
                                log.warning(f"无法解析 SSE 数据: {data}")
                                continue
                except Exception as e_chunk:
                    log.error(f"处理异步流式数据块时出错: {str(e_chunk)}")
                    yield f"处理响应数据块时出错: {str(e_chunk)}"
        except Exception as e_stream:
            log.error(f"异步流式响应处理出错: {str(e_stream)}")
            yield f"异步流式响应处理出错: {str(e_stream)}"
