import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
# google.generativeai removed - using OpenAI-compatible providers only
import asyncio

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化调度器
    try:
        from api.scheduler import get_scheduler
        get_scheduler()
        logger.info("定时任务调度器已启动")
    except Exception as e:
        logger.warning(f"调度器启动失败（非致命）: {e}")
    yield
    # 关闭时停止调度器
    try:
        from api.scheduler import shutdown_scheduler
        shutdown_scheduler()
    except Exception:
        pass


# 初始化 FastAPI 应用
app = FastAPI(
    title="Streaming API",
    description="流式聊天补全 API",
    lifespan=lifespan,
)

# 配置跨域资源共享（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 辅助函数：获取 adalflow 默认根路径
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic 数据模型 ---
class WikiPage(BaseModel):
    """
    Wiki 页面数据模型。
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Wiki 章节数据模型。
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    整体 Wiki 结构数据模型。
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Wiki 缓存存储数据模型。
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    保存 Wiki 缓存的请求体数据模型。
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Wiki 导出请求数据模型。
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- 模型配置数据模型 ---
class Model(BaseModel):
    """
    LLM 模型配置数据模型。
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    LLM 提供商配置数据模型。
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    完整模型配置数据模型。
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE, get_provider_base_url, set_provider_base_url, runtime_overrides

@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]

@app.get("/auth/status")
async def get_auth_status():
    """
    检查 Wiki 是否需要身份验证。
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    校验授权码是否正确。
    """
    return {"success": WIKI_AUTH_CODE == request.code}

class ProviderUrlUpdate(BaseModel):
    provider: str = Field(..., description="Provider ID (e.g., vllm, deepseek, ollama)")
    base_url: str = Field(..., description="New base URL for the provider")

class CustomModelAdd(BaseModel):
    provider: str = Field(..., description="Provider ID")
    model_id: str = Field(..., description="Model identifier")
    temperature: Optional[float] = Field(0.7, description="Temperature")
    top_p: Optional[float] = Field(0.8, description="Top P")

@app.get("/models/runtime_config")
async def get_runtime_config():
    """获取当前运行时配置，包括各提供商的 base URL。"""
    provider_urls = {}
    for provider_id in configs.get("providers", {}).keys():
        provider_urls[provider_id] = get_provider_base_url(provider_id)
    return {"provider_urls": provider_urls}

@app.post("/models/provider_url")
async def update_provider_url(request: ProviderUrlUpdate):
    """在运行时更新指定提供商的 base URL。"""
    set_provider_base_url(request.provider, request.base_url)
    return {"success": True, "provider": request.provider, "base_url": request.base_url}

PRESETS_FILE = os.environ.get(
    "MODEL_PRESETS_FILE",
    os.path.join(os.path.expanduser("~"), ".adalflow", "model_presets.json")
)

def _load_presets() -> list:
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_presets(presets: list):
    os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

class ModelPreset(BaseModel):
    name: str = Field(..., description="Model name/identifier")
    provider: str = Field(..., description="Provider ID")
    base_url: Optional[str] = Field(None, description="API Base URL (optional)")
    api_key: Optional[str] = Field(None, description="API Key (optional)")

@app.get("/models/presets")
async def get_model_presets():
    """获取管理员维护的模型预设列表。"""
    return {"presets": _load_presets()}

@app.post("/models/presets")
async def add_model_preset(preset: ModelPreset):
    """添加一个模型预设。"""
    presets = _load_presets()
    presets.append(preset.dict())
    _save_presets(presets)
    return {"success": True, "presets": presets}

@app.delete("/models/presets/{index}")
async def delete_model_preset(index: int):
    """按索引删除一个模型预设。"""
    presets = _load_presets()
    if index < 0 or index >= len(presets):
        raise HTTPException(status_code=404, detail="Preset index out of range")
    presets.pop(index)
    _save_presets(presets)
    return {"success": True, "presets": presets}

@app.post("/models/add_model")
async def add_custom_model(request: CustomModelAdd):
    """在运行时向指定提供商的模型列表中添加自定义模型。"""
    if "providers" not in configs:
        raise HTTPException(status_code=500, detail="Provider configuration not loaded")
    provider_config = configs["providers"].get(request.provider)
    if not provider_config:
        raise HTTPException(status_code=404, detail=f"Provider '{request.provider}' not found")
    # Add the model to the provider's models dict
    model_params = {"temperature": request.temperature or 0.7, "top_p": request.top_p or 0.8}
    if request.provider == "ollama":
        model_params = {"options": {"temperature": request.temperature or 0.7, "top_p": request.top_p or 0.8, "num_ctx": 32000}}
    provider_config["models"][request.model_id] = model_params
    return {"success": True, "provider": request.provider, "model_id": request.model_id}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    获取可用的模型提供商及其模型列表。

    返回应用中所有可用的模型提供商及其对应模型的配置信息。

    Returns:
        ModelConfig: 包含提供商和模型列表的配置对象。
    """
    try:
        logger.info("正在获取模型配置")

        # 根据配置文件构建提供商列表
        providers = []
        default_provider = configs.get("default_provider", "google")

        # 遍历 config.py 中的提供商配置
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # 从配置中添加模型
            for model_id in provider_config["models"].keys():
                # 尽可能使用更友好的显示名称
                models.append(Model(id=model_id, name=model_id))

            # 添加提供商及其模型
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="vllm",
                    name="Vllm",
                    supportsCustomModel=True,
                    models=[
                        Model(id="QWQ3-32b", name="QWQ3-32b")
                    ]
                )
            ],
            defaultProvider="vllm"
        )

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"正在导出 {request.repo_url} 的 Wiki，格式：{request.format}")

        # 从 URL 中提取仓库名称，用于生成文件名
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # 获取当前时间戳，用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # 生成 Markdown 格式内容
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON 格式
            # 生成 JSON 格式内容
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # 创建包含文件下载头的响应
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"导出 Wiki 时出错: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload/repo")
async def upload_repo_zip(file: UploadFile = File(...)):
    """
    上传包含源代码的 ZIP 文件。
    解压到服务器临时目录后返回路径，供 /local_repo/structure 接口使用。
    解决了「本地路径」仅存在于用户机器上、服务器无法访问的问题。
    """
    import tempfile
    import zipfile
    import shutil

    if not file.filename or not file.filename.lower().endswith('.zip'):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a .zip file"}
        )

    tmp_dir = None
    try:
        # 创建持久化上传目录（存储在 volume 挂载目录中，重建容器后不丢失）
        from adalflow.utils import get_adalflow_default_root_path
        upload_base = os.path.join(get_adalflow_default_root_path(), "uploads")
        os.makedirs(upload_base, exist_ok=True)

        # 根据上传文件名生成唯一目录名
        import re as _re
        safe_name = _re.sub(r'[^\w\-.]', '_', file.filename.rsplit('.', 1)[0])
        tmp_dir = os.path.join(upload_base, f"{safe_name}_{int(datetime.now().timestamp())}")
        os.makedirs(tmp_dir, exist_ok=True)

        # 保存上传的文件
        zip_path = os.path.join(tmp_dir, file.filename)
        with open(zip_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # 解压 ZIP 文件
        extract_dir = os.path.join(tmp_dir, "repo")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 解压完成后删除 ZIP 文件以节省空间
        os.remove(zip_path)

        # 如果 ZIP 内只有一个顶层目录，将其作为仓库根目录
        entries = os.listdir(extract_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            repo_path = os.path.join(extract_dir, entries[0])
        else:
            repo_path = extract_dir

        logger.info(f"上传的仓库已解压到: {repo_path}")
        return {"path": repo_path, "message": "上传成功"}

    except zipfile.BadZipFile:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid ZIP file"}
        )
    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"Error processing uploaded repo: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing upload: {str(e)}"}
        )

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="本地仓库路径")):
    """返回本地仓库的文件树结构及 README 内容。"""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found on server: {path}. "
                     "Note: 'Local path' must be a path on the server machine, not your browser machine. "
                     "If the code is on your local computer, please use the 'Upload ZIP' feature to upload it first."}
        )

    try:
        logger.info(f"正在处理本地仓库: {path}")
        file_tree_lines = []
        readme_content = ""

        # 从 repo.json 配置中读取排除目录列表
        repo_config = configs.get("repo_config", {}).get("file_filters", {})
        excluded_dir_names = set()
        for d in repo_config.get("excluded_dirs", []):
            # 从 "./.venv/"、"./node_modules/" 等模式中提取目录名
            name = d.strip('./').strip('/')
            if name:
                excluded_dir_names.add(name)
        # 始终排除以下常见非代码目录
        excluded_dir_names.update({
            '.git', '.svn', '.hg', '.bzr', '__pycache__', 'node_modules',
            '.venv', 'venv', 'env', 'virtualenv', 'bower_components',
            'jspm_packages', 'dist', 'build', 'out', 'bin', 'obj',
            'target', '.idea', '.vscode', '.vs', 'coverage', 'htmlcov',
            '.tox', '.nyc_output', '.output', 'bld', 'lib-cov',
            '.next', '.nuxt', '.cache', '.parcel-cache', 'tmp', 'temp',
            '.gradle', '.mvn', 'vendor', 'packages',
        })

        # 从 repo.json 构建排除文件扩展名集合
        import fnmatch
        excluded_file_patterns = repo_config.get("excluded_files", [])
        excluded_extensions = set()
        for p in excluded_file_patterns:
            if p.startswith('*.'):
                excluded_extensions.add(p[1:])  # 如 '.min.js'

        # 始终排除二进制/非代码文件扩展名
        excluded_extensions.update({
            '.db', '.sqlite', '.sqlite3', '.pkl', '.pickle', '.npy', '.npz',
            '.h5', '.hdf5', '.parquet', '.feather', '.arrow',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
            '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flv',
            '.woff', '.woff2', '.ttf', '.otf', '.eot',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.log',
        })

        MAX_FILES = 5000  # 文件数量安全上限

        for root, dirs, files in os.walk(path):
            # 按目录名排除（跳过隐藏目录）
            dirs[:] = [d for d in dirs if d not in excluded_dir_names and not d.startswith('.')]

            for file in files:
                if len(file_tree_lines) >= MAX_FILES:
                    break
                if file.startswith('.') or file == '.DS_Store':
                    continue
                # 检查文件扩展名
                _, ext = os.path.splitext(file.lower())
                if ext in excluded_extensions:
                    continue
                # 检查文件名模式
                skip = False
                for pattern in excluded_file_patterns:
                    if not pattern.startswith('*.') and fnmatch.fnmatch(file, pattern):
                        skip = True
                        break
                if skip:
                    continue

                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file.replace('\\', '/'))
                # 查找 README.md（大小写不敏感）
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='replace') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

            if len(file_tree_lines) >= MAX_FILES:
                logger.warning(f"File tree truncated at {MAX_FILES} files for {path}")
                break

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        logger.info(f"本地仓库结构: {len(file_tree_lines)} 个文件，{len(file_tree_str)} 个字符")
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"处理本地仓库时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"处理本地仓库时出错: {str(e)}"}
        )

@app.get("/repo/structure")
async def get_repo_structure(
    repo_url: str = Query(..., description="仓库 URL"),
    repo_type: str = Query("gitea", description="仓库类型（gitea、gitee、github、gitlab、svn 等）"),
    token: str = Query(None, description="私有仓库的访问令牌"),
    svn_username: str = Query(None, description="SVN 用户名"),
    svn_password: str = Query(None, description="SVN 密码")
):
    """
    通过在后端克隆仓库来获取仓库文件结构。
    用于 Gitea/Gitee/SVN 等远程仓库，以避免浏览器 CORS 问题。
    返回文件树和 README 内容，支持 Git 和 SVN 两种仓库类型。
    """
    import tempfile
    import shutil
    import subprocess

    # Detect if this is an SVN repository
    is_svn = (repo_type == "svn" or "/svn/" in repo_url.lower())

    tmp_dir = None
    try:
        logger.info(f"正在获取远程仓库结构: {repo_url}（类型={repo_type}，is_svn={is_svn}）")

        # 创建用于克隆/检出的临时目录
        tmp_dir = tempfile.mkdtemp(prefix="deepwiki_repo_")

        if is_svn:
            # --- SVN 检出 ---
            svn_cmd = ["svn", "checkout", "--depth", "infinity"]

            # 若提供了认证信息则附加到命令中
            if svn_username and svn_password:
                svn_cmd.extend(["--username", svn_username, "--password", svn_password, "--non-interactive", "--trust-server-cert"])
            elif token:
                # 以 token 作为密码，用户名留空
                svn_cmd.extend(["--username", "", "--password", token, "--non-interactive", "--trust-server-cert"])
            else:
                svn_cmd.extend(["--non-interactive", "--trust-server-cert"])

            svn_cmd.extend([repo_url, tmp_dir])

            result = subprocess.run(
                svn_cmd,
                capture_output=True, text=True, timeout=180
            )

            if result.returncode != 0:
                error_msg = result.stderr
                # 从错误信息中清除敏感凭据
                if svn_password:
                    error_msg = error_msg.replace(svn_password, '***')
                if token:
                    error_msg = error_msg.replace(token, '***')

                # checkout 失败时降级尝试 svn list（允许列目录但不允许检出的场景）
                logger.warning(f"SVN checkout 失败，尝试 svn list: {error_msg}")
                list_cmd = ["svn", "list", "-R"]
                if svn_username and svn_password:
                    list_cmd.extend(["--username", svn_username, "--password", svn_password, "--non-interactive", "--trust-server-cert"])
                elif token:
                    list_cmd.extend(["--username", "", "--password", token, "--non-interactive", "--trust-server-cert"])
                else:
                    list_cmd.extend(["--non-interactive", "--trust-server-cert"])
                list_cmd.append(repo_url)

                list_result = subprocess.run(
                    list_cmd,
                    capture_output=True, text=True, timeout=120
                )

                if list_result.returncode != 0:
                    list_error = list_result.stderr
                    if svn_password:
                        list_error = list_error.replace(svn_password, '***')
                    if token:
                        list_error = list_error.replace(token, '***')
                    raise Exception(f"SVN checkout 和 list 均失败。checkout 错误: {error_msg}。list 错误: {list_error}")

                # 解析 svn list 输出（每行一个文件）
                file_tree_lines = []
                excluded_dirs = {'.svn', '__pycache__', 'node_modules', '.venv', 'venv',
                                 'dist', 'build', '.idea', '.vscode', 'target', 'bin', 'obj'}
                for line in list_result.stdout.strip().split('\n'):
                    line = line.strip()
                    if not line or line.endswith('/'):
                        # 跳过目录条目（目录以 / 结尾）
                        # But check if any excluded dir is in the path
                        continue
                    # Check if path contains excluded directories
                    parts = line.split('/')
                    skip = False
                    for part in parts[:-1]:  # Check directory parts only
                        if part in excluded_dirs or part.startswith('.'):
                            skip = True
                            break
                    if skip:
                        continue
                    # Skip hidden files
                    filename = parts[-1]
                    if filename.startswith('.'):
                        continue
                    file_tree_lines.append(line)

                file_tree_str = '\n'.join(sorted(file_tree_lines))
                return {"file_tree": file_tree_str, "readme": ""}

        else:
            # --- Git 浅克隆 ---
            from urllib.parse import urlparse, urlunparse, quote
            parsed = urlparse(repo_url)
            clone_url = repo_url
            if token:
                encoded_token = quote(token, safe='')
                if repo_type == "gitlab":
                    clone_url = urlunparse((parsed.scheme, f"oauth2:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
                elif repo_type == "bitbucket":
                    clone_url = urlunparse((parsed.scheme, f"x-token-auth:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
                else:
                    # github、gitea、gitee 均使用 token@host 格式
                    clone_url = urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))

            # 浅克隆（depth=1）以加快速度
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, tmp_dir],
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                error_msg = result.stderr.replace(token, '***') if token else result.stderr
                raise Exception(f"Git 克隆失败: {error_msg}")

        # 遍历克隆/检出后的仓库，构建文件树并提取 README
        file_tree_lines = []
        readme_content = ""
        excluded_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.svn', '.hg',
                         'dist', 'build', '.idea', '.vscode', '.vs', 'target', 'bin', 'obj'}

        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
            for file in files:
                if file.startswith('.') or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, tmp_dir)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file.replace('\\', '/'))
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='replace') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"无法读取 README.md: {str(e)}")

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}

    except Exception as e:
        logger.error(f"获取仓库结构时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"获取仓库结构时出错: {str(e)}"}
        )
    finally:
        # 清理临时目录
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


class RepoFilesRequest(BaseModel):
    repo_url: str = Field(..., description="Repository URL or local path")
    type: str = Field("local", description="Repository type")
    file_paths: List[str] = Field(..., description="List of file paths to read")
    token: Optional[str] = Field(None, description="Access token for private repos")

@app.post("/repo/files")
async def get_repo_files(request: RepoFilesRequest):
    """
    读取仓库中指定文件的内容。
    本地仓库直接读取；远程仓库使用缓存克隆目录。
    返回文件路径到文件内容的映射字典。
    """
    import tempfile
    import shutil

    results = {}
    repo_path = None
    tmp_dir = None

    try:
        if request.type == "local":
            repo_path = request.repo_url
            if not os.path.isdir(repo_path):
                raise HTTPException(status_code=404, detail=f"本地目录不存在: {repo_path}")
        else:
            # 远程仓库：克隆到临时目录（浅克隆）
            tmp_dir = tempfile.mkdtemp(prefix="deepwiki_files_")
            from urllib.parse import urlparse, urlunparse, quote
            parsed = urlparse(request.repo_url)
            clone_url = request.repo_url
            if request.token:
                encoded_token = quote(request.token, safe='')
                clone_url = urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))

            import subprocess
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, tmp_dir],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                error_msg = result.stderr.replace(request.token, '***') if request.token else result.stderr
                raise Exception(f"Git 克隆失败: {error_msg}")
            repo_path = tmp_dir

        # 读取请求的文件内容
        MAX_FILE_SIZE = 100_000  # 单文件上限 100KB
        MAX_TOTAL_SIZE = 500_000  # 总大小上限 500KB
        total_size = 0

        for file_path in request.file_paths:
            if total_size >= MAX_TOTAL_SIZE:
                break
            # 规范化路径，防止目录遍历攻击
            safe_path = os.path.normpath(file_path).lstrip(os.sep).lstrip('/')
            full_path = os.path.join(repo_path, safe_path)

            if not os.path.isfile(full_path):
                results[file_path] = None
                continue

            file_size = os.path.getsize(full_path)
            if file_size > MAX_FILE_SIZE:
                results[file_path] = f"[文件过大: {file_size} 字节，已跳过]"
                continue

            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                results[file_path] = content
                total_size += len(content)
            except Exception as e:
                results[file_path] = f"[读取文件时出错: {str(e)}]"

        return {"files": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取仓库文件时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    将 wiki 页面生成 Markdown 格式的导出内容。

    Args:
        repo_url: 仓库 URL
        pages: wiki 页面列表

    Returns:
        Markdown 格式的字符串内容
    """
    # 从元数据开始构建 Markdown 文档
    markdown = f"# {repo_url} 的 Wiki 文档\n\n"
    markdown += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # 添加目录
    markdown += "## 目录\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "相关主题: " + ", ".join(related_titles) + "\n\n"

        # 添加页面正文内容
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    将 wiki 页面生成 JSON 格式的导出内容。

    Args:
        repo_url: 仓库 URL
        pages: wiki 页面列表

    Returns:
        JSON 格式的字符串内容
    """
    # 创建包含元数据和页面内容的字典
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # 格式化为 JSON 字符串并返回
    return json.dumps(export_data, indent=2)

# 导入简化版聊天实现
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat

# 将 chat_completions_stream 端点注册到主应用
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# 注册 WebSocket 端点
app.add_api_websocket_route("/ws/chat", handle_websocket_chat)

# --- 直接 LLM 流式端点（无 RAG）---
# 用于 wiki 结构生成，此时文件树已包含在 prompt 中

from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from api.openai_client import OpenAIClient
from api.config import get_model_config as get_model_config_func, VLLM_API_KEY, VLLM_BASE_URL, get_provider_base_url as get_base_url

class DirectChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="聊天消息列表")
    provider: str = Field("deepseek", description="模型提供商")
    model: Optional[str] = Field(None, description="模型名称")
    api_key: Optional[str] = Field(None, description="可选的 API Key 覆盖")

@app.post("/chat/direct/stream")
async def chat_direct_stream(request: DirectChatRequest):
    """
    直接 LLM 流式端点，不使用 RAG。
    用于 wiki 结构生成，此时文件树已包含在 prompt 中。
    """
    import asyncio

    try:
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1]
        prompt = last_message.get("content", "")

        model_config = get_model_config_func(request.provider, request.model)["model_kwargs"]

        # 解析 API Key：优先使用请求中的覆盖值，否则回退到环境变量
        request_api_key = request.api_key.strip() if request.api_key and request.api_key.strip() else None

        if request.provider == "ollama":
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
        elif request.provider == "vllm":
            vllm_url = get_base_url("vllm")
            vllm_key = request_api_key or VLLM_API_KEY
            model = OpenAIClient(api_key=vllm_key, base_url=vllm_url)
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
        else:
            # DeepSeek 或其他 OpenAI 兼容提供商
            provider_url = get_base_url(request.provider)
            if request_api_key:
                model = OpenAIClient(api_key=request_api_key, base_url=provider_url)
            else:
                model = OpenAIClient(base_url=provider_url)
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )

        def strip_think_tags(text: str) -> str:
            """去除推理模型输出中的 <think> 和 </think> 标签。"""
            import re as _re
            # Remove <think>...</think> blocks entirely (including content)
            cleaned = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL)
            # Also remove orphaned opening/closing tags (for streaming chunks)
            cleaned = cleaned.replace('<think>', '').replace('</think>', '')
            return cleaned

        async def generate():
            try:
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                if request.provider == "ollama":
                    async for chunk in response:
                        text = None
                        if isinstance(chunk, dict):
                            text = chunk.get("message", {}).get("content") if isinstance(chunk.get("message"), dict) else chunk.get("message")
                        else:
                            message = getattr(chunk, "message", None)
                            if message is not None:
                                text = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
                        if not text:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None)
                        if isinstance(text, str) and text:
                            clean_text = text.replace('<think>', '').replace('</think>', '')
                            yield clean_text
                else:
                    async for chunk in response:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text = getattr(delta, "content", None)
                                if text is not None:
                                    # 对所有提供商去除 think 标签（如 vllm 下的 QWQ 推理模型）
                                    clean_text = text.replace('<think>', '').replace('</think>', '')
                                    yield clean_text
            except Exception as e:
                logger.error(f"直接 LLM 流式响应出错: {str(e)}")
                yield f"\n[STREAM_ERROR] {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"chat_direct_stream 出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)  # 确保缓存目录存在

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """生成指定仓库 wiki 缓存文件的路径。"""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """从文件系统读取 wiki 缓存数据。"""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"从 {cache_path} 读取 wiki 缓存时出错: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """将 wiki 缓存数据保存到文件系统。"""
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model
        )
        # 记录待缓存数据的大小（避免日志中输出完整内容）
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"缓存载荷已准备就绪，大小: {payload_size} 字节。")
        except Exception as ser_e:
            logger.warning(f"无法序列化载荷以记录大小: {ser_e}")


        logger.info(f"正在将缓存文件写入: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki 缓存已成功保存到 {cache_path}")
        return True
    except IOError as e:
        logger.error(f"保存 Wiki 缓存到 {cache_path} 时发生 IOError: {e.strerror}（errno: {e.errno}）", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"保存 Wiki 缓存到 {cache_path} 时发生意外错误: {e}", exc_info=True)
        return False

# --- Wiki 缓存 API 端点 ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="仓库所有者"),
    repo: str = Query(..., description="仓库名称"),
    repo_type: str = Query(..., description="仓库类型（如 github、gitlab）"),
    language: str = Query(..., description="Wiki 内容的语言")
):
    """
    获取指定仓库的 Wiki 缓存数据（结构和已生成页面）。
    """
    # 语言合法性校验
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"正在尝试获取 {owner}/{repo}（{repo_type}）的 Wiki 缓存，语言: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # 未找到缓存时返回 200 + null，前端依赖此行为
        logger.info(f"未找到 {owner}/{repo}（{repo_type}）的 Wiki 缓存，语言: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    将生成的 Wiki 数据（结构和页面）保存到服务端缓存。
    """
    # 语言合法性校验
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"正在尝试保存 {request_data.repo.owner}/{request_data.repo.repo}（{request_data.repo.type}）的 Wiki 缓存，语言: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki 缓存已成功保存"}
    else:
        raise HTTPException(status_code=500, detail="保存 Wiki 缓存失败")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="仓库所有者"),
    repo: str = Query(..., description="仓库名称"),
    repo_type: str = Query(..., description="仓库类型（如 github、gitlab）"),
    language: str = Query(..., description="Wiki 内容的语言"),
    authorization_code: Optional[str] = Query(None, description="授权码")
):
    """
    从文件系统中删除指定的 Wiki 缓存。
    """
    # 语言合法性校验
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="不支持该语言")

    if WIKI_AUTH_MODE:
        logger.info("正在验证授权码")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="授权码无效")

    logger.info(f"正在尝试删除 {owner}/{repo}（{repo_type}）的 Wiki 缓存，语言: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"已成功删除 Wiki 缓存: {cache_path}")
            return {"message": f"已成功删除 {owner}/{repo}（{language}）的 Wiki 缓存"}
        except Exception as e:
            logger.error(f"删除 Wiki 缓存 {cache_path} 时出错: {e}")
            raise HTTPException(status_code=500, detail=f"删除 Wiki 缓存失败: {str(e)}")
    else:
        logger.warning(f"Wiki 缓存不存在，无法删除: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki 缓存不存在")

@app.get("/health")
async def health_check():
    """健康检查端点，用于 Docker 和监控系统。"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.get("/")
async def root():
    """根路由端点，检查 API 运行状态并动态列出所有可用端点。"""
    # 动态收集 FastAPI 应用中注册的所有路由
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # 跳过文档和静态路由
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # Use asyncio.to_thread for os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # Use asyncio.to_thread for os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # Expecting repo_type_owner_repo_language
                    # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en.json
                    # parts = [github, AsyncFuncAI, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1] # language is the last part
                        repo = "_".join(parts[2:-1]) # repo can contain underscores

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000), # Convert to milliseconds
                                language=language
                            )
                        )
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")


# ─────────────────────────────────────────────
# 定时拉取调度任务 API
# ─────────────────────────────────────────────

class ScheduleCreateRequest(BaseModel):
    name: str
    repo_url: str
    repo_type: str = "gitlab"
    access_token: str = ""
    interval_hours: int = 24
    cron_expr: str = ""
    enabled: bool = True
    excluded_dirs: str = ""
    excluded_files: str = ""


class ScheduleUpdateRequest(BaseModel):
    name: Optional[str] = None
    repo_url: Optional[str] = None
    repo_type: Optional[str] = None
    access_token: Optional[str] = None
    interval_hours: Optional[int] = None
    cron_expr: Optional[str] = None
    enabled: Optional[bool] = None
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None


@app.get("/api/schedules")
async def list_schedules_endpoint():
    """列出所有定时拉取任务。"""
    try:
        from api.scheduler import list_schedules
        return list_schedules()
    except Exception as e:
        logger.error(f"列出调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedules", status_code=201)
async def create_schedule_endpoint(req: ScheduleCreateRequest):
    """创建新的定时拉取任务。"""
    try:
        from api.scheduler import create_schedule
        return create_schedule(req.model_dump())
    except Exception as e:
        logger.error(f"创建调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/schedules/{schedule_id}")
async def update_schedule_endpoint(schedule_id: str, req: ScheduleUpdateRequest):
    """更新指定调度任务。"""
    try:
        from api.scheduler import update_schedule
        data = {k: v for k, v in req.model_dump().items() if v is not None}
        result = update_schedule(schedule_id, data)
        if result is None:
            raise HTTPException(status_code=404, detail="Schedule not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule_endpoint(schedule_id: str):
    """删除指定调度任务。"""
    try:
        from api.scheduler import delete_schedule
        ok = delete_schedule(schedule_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Schedule not found")
        return {"message": "Deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedules/{schedule_id}/trigger")
async def trigger_schedule_endpoint(schedule_id: str):
    """立即手动触发一次调度任务。"""
    try:
        from api.scheduler import get_schedule, trigger_schedule_now
        cfg = get_schedule(schedule_id)
        if cfg is None:
            raise HTTPException(status_code=404, detail="Schedule not found")
        trigger_schedule_now(schedule_id)
        return {"message": "Triggered"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"手动触发调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
