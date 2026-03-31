import logging
import os
from typing import List, Optional
from urllib.parse import unquote

from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENAI_API_KEY, VLLM_API_KEY, VLLM_BASE_URL
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.rag import RAG
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT
)

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# 初始化 FastAPI 应用
app = FastAPI(
    title="Simple Chat API",
    description="用于流式聊天补全的简化 API"
)

# 配置跨域资源共享（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# API 数据模型
class ChatMessage(BaseModel):
    role: str  # 'user'（用户）或 'assistant'（助手）
    content: str

class ChatCompletionRequest(BaseModel):
    """
    聊天补全请求数据模型。
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("deepseek", description="Model provider (deepseek, vllm, ollama)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """流式返回聊天补全响应（基于 OpenAI 兼容接口）。"""
    try:
        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom included files: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                raise HTTPException(status_code=500, detail="No valid document embeddings found. This may be due to embedding size inconsistencies or API errors during document processing. Please try again or check your repository content.")
            else:
                logger.error(f"ValueError preparing retriever: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error preparing retriever: {str(e)}")
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            # Check for specific embedding-related errors
            if "All embeddings should be of the same size" in str(e):
                raise HTTPException(status_code=500, detail="Inconsistent embedding sizes detected. Some documents may have failed to embed properly. Please try again.")
            else:
                raise HTTPException(status_code=500, detail=f"Error preparing retriever: {str(e)}")

        # Validate request
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from the user")

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None

        if not input_too_large:
            try:
                # If filePath exists, modify the query for RAG to focus on the file
                rag_query = query
                if request.filePath:
                    # Use the file path to get relevant context about the file
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")

                # Try to perform RAG retrieval
                try:
                    # This will use the actual RAG implementation
                    retrieved_documents = request_rag(rag_query, language=request.language)

                    if retrieved_documents and retrieved_documents[0].documents:
                        # Format context for the prompt in a more structured way
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")

                        # Group documents by file path
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # Format context text with file path grouping
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # Add file header with metadata
                            header = f"## File Path: {file_path}\n\n"
                            # Add document content
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # Join all parts with clear separation
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("No documents retrieved from RAG")
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")
                    # Continue without RAG if there's an error

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            elif is_final_iteration:
                system_prompt = DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
            else:
                system_prompt = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Create the prompt with context
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

        # Check if filePath is provided and fetch file content if it exists
        if file_content:
            # Add file content to the prompt after conversation history
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # Only include context if it's not empty
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # Add a note that we're skipping RAG due to size constraints or because it's the isolated API
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "ollama":
            prompt += " /no_think"

            model = OllamaClient()
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "options": {
                    "temperature": model_config.get("temperature", 0.7),
                    "top_p": model_config.get("top_p", 0.8),
                    "num_ctx": model_config.get("num_ctx", 32000)
                }
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "vllm":
            logger.info(f"Using vLLM with model: {request.model}")

            model = OpenAIClient(
                api_key=VLLM_API_KEY,
                base_url=VLLM_BASE_URL,
            )
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config.get("temperature", 0.7)
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        else:
            # DeepSeek or any OpenAI-compatible provider (default)
            logger.info(f"Using {request.provider} with model: {request.model}")

            model = OpenAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config.get("temperature", 0.7)
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )

        # Create a streaming response
        async def response_stream():
            try:
                if request.provider == "ollama":
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    async for chunk in response:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                        if text and not text.startswith('model=') and not text.startswith('created_at='):
                            text = text.replace('<think>', '').replace('</think>', '')
                            yield text
                else:
                    # OpenAI-compatible providers (deepseek, vllm)
                    try:
                        logger.info(f"Making {request.provider} API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        async for chunk in response:
                            choices = getattr(chunk, "choices", [])
                            if len(choices) > 0:
                                delta = getattr(choices[0], "delta", None)
                                if delta is not None:
                                    text = getattr(delta, "content", None)
                                    if text is not None:
                                        yield text
                    except Exception as e_api:
                        logger.error(f"Error with {request.provider} API: {str(e_api)}")
                        yield f"\nError with {request.provider} API: {str(e_api)}\n\nPlease check your API configuration."

            except Exception as e_outer:
                logger.error(f"Error in streaming response: {str(e_outer)}")
                error_message = str(e_outer)

                if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                    logger.warning("Token limit exceeded, retrying without context")
                    try:
                        simplified_prompt = f"/no_think {system_prompt}\n\n"
                        if conversation_history:
                            simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
                        if request.filePath and file_content:
                            simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"
                        simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\n\n"
                        simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                        if request.provider == "ollama":
                            simplified_prompt += " /no_think"

                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=model_kwargs,
                            model_type=ModelType.LLM
                        )
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                        if request.provider == "ollama":
                            async for chunk in fallback_response:
                                text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                                if text and not text.startswith('model=') and not text.startswith('created_at='):
                                    text = text.replace('<think>', '').replace('</think>', '')
                                    yield text
                        else:
                            async for chunk in fallback_response:
                                choices = getattr(chunk, "choices", [])
                                if len(choices) > 0:
                                    delta = getattr(choices[0], "delta", None)
                                    if delta is not None:
                                        text = getattr(delta, "content", None)
                                        if text is not None:
                                            yield text
                    except Exception as e2:
                        logger.error(f"Error in fallback streaming response: {str(e2)}")
                        yield f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts."
                else:
                    yield f"\nError: {error_message}"

        # Return streaming response
        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"Error in streaming chat completion: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """根路由，用于检查 API 是否正常运行。"""
    return {"status": "API is running", "message": "Navigate to /docs for API documentation"}
