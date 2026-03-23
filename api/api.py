import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
# google.generativeai removed - using OpenAI-compatible providers only
import asyncio

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
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
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE

@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]

@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # Add models from config
            for model_id in provider_config["models"].keys():
                # Get a more user-friendly display name if possible
                models.append(Model(id=model_id, name=model_id))

            # Add provider with its models
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
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash")
                    ]
                )
            ],
            defaultProvider="google"
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
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        # Use comprehensive exclusion list from repo.json config
        repo_config = configs.get("repo_config", {}).get("file_filters", {})
        excluded_dir_names = set()
        for d in repo_config.get("excluded_dirs", []):
            # Extract directory name from patterns like "./.venv/", "./node_modules/"
            name = d.strip('./').strip('/')
            if name:
                excluded_dir_names.add(name)
        # Always exclude these common non-code directories
        excluded_dir_names.update({
            '.git', '.svn', '.hg', '.bzr', '__pycache__', 'node_modules',
            '.venv', 'venv', 'env', 'virtualenv', 'bower_components',
            'jspm_packages', 'dist', 'build', 'out', 'bin', 'obj',
            'target', '.idea', '.vscode', '.vs', 'coverage', 'htmlcov',
            '.tox', '.nyc_output', '.output', 'bld', 'lib-cov',
            '.next', '.nuxt', '.cache', '.parcel-cache', 'tmp', 'temp',
            '.gradle', '.mvn', 'vendor', 'packages',
        })

        # Build set of excluded file extensions from repo.json
        import fnmatch
        excluded_file_patterns = repo_config.get("excluded_files", [])
        excluded_extensions = set()
        for p in excluded_file_patterns:
            if p.startswith('*.'):
                excluded_extensions.add(p[1:])  # e.g., '.min.js'

        # Binary/non-code extensions to always exclude
        excluded_extensions.update({
            '.db', '.sqlite', '.sqlite3', '.pkl', '.pickle', '.npy', '.npz',
            '.h5', '.hdf5', '.parquet', '.feather', '.arrow',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
            '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flv',
            '.woff', '.woff2', '.ttf', '.otf', '.eot',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.log',
        })

        MAX_FILES = 5000  # Safety limit

        for root, dirs, files in os.walk(path):
            # Exclude dirs by name
            dirs[:] = [d for d in dirs if d not in excluded_dir_names and not d.startswith('.')]

            for file in files:
                if len(file_tree_lines) >= MAX_FILES:
                    break
                if file.startswith('.') or file == '.DS_Store':
                    continue
                # Check extension
                _, ext = os.path.splitext(file.lower())
                if ext in excluded_extensions:
                    continue
                # Check filename patterns
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
                # Find README.md (case-insensitive)
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
        logger.info(f"Local repo structure: {len(file_tree_lines)} files, {len(file_tree_str)} chars")
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

@app.get("/repo/structure")
async def get_repo_structure(
    repo_url: str = Query(..., description="Repository URL"),
    repo_type: str = Query("gitea", description="Repository type (gitea, gitee, github, gitlab, etc.)"),
    token: str = Query(None, description="Access token for private repositories")
):
    """
    Fetch repository structure by cloning the repo on the backend.
    Used for Gitea/Gitee/any remote repos to avoid CORS issues in the browser.
    Returns file tree and README content.
    """
    import tempfile
    import shutil

    tmp_dir = None
    try:
        logger.info(f"Fetching remote repo structure: {repo_url} (type={repo_type})")

        # Create a temporary directory for cloning
        tmp_dir = tempfile.mkdtemp(prefix="deepwiki_repo_")

        # Build clone URL with token if provided
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
                # github, gitea, gitee all use token@host format
                clone_url = urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))

        # Shallow clone (depth=1) for speed
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, tmp_dir],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            error_msg = result.stderr.replace(token, '***') if token else result.stderr
            raise Exception(f"Git clone failed: {error_msg}")

        # Walk the cloned repo to get file tree and README
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
                        logger.warning(f"Could not read README.md: {str(e)}")

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}

    except Exception as e:
        logger.error(f"Error fetching repo structure: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching repo structure: {str(e)}"}
        )
    finally:
        # Clean up temp directory
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
    Read specific file contents from a repository.
    For local repos, reads directly. For remote repos, uses cached clone.
    Returns a dict mapping file paths to their contents.
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
                raise HTTPException(status_code=404, detail=f"Local directory not found: {repo_path}")
        else:
            # For remote repos, clone to temp dir (shallow)
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
                raise Exception(f"Git clone failed: {error_msg}")
            repo_path = tmp_dir

        # Read requested files
        MAX_FILE_SIZE = 100_000  # 100KB per file
        MAX_TOTAL_SIZE = 500_000  # 500KB total
        total_size = 0

        for file_path in request.file_paths:
            if total_size >= MAX_TOTAL_SIZE:
                break
            # Sanitize path to prevent directory traversal
            safe_path = os.path.normpath(file_path).lstrip(os.sep).lstrip('/')
            full_path = os.path.join(repo_path, safe_path)

            if not os.path.isfile(full_path):
                results[file_path] = None
                continue

            file_size = os.path.getsize(full_path)
            if file_size > MAX_FILE_SIZE:
                results[file_path] = f"[File too large: {file_size} bytes, skipped]"
                continue

            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                results[file_path] = content
                total_size += len(content)
            except Exception as e:
                results[file_path] = f"[Error reading file: {str(e)}]"

        return {"files": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading repo files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
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
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_api_websocket_route("/ws/chat", handle_websocket_chat)

# --- Direct LLM streaming endpoint (no RAG) ---
# Used for wiki structure determination where file tree is already in the prompt

from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from api.openai_client import OpenAIClient
from api.config import get_model_config as get_model_config_func, VLLM_API_KEY, VLLM_BASE_URL

class DirectChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of chat messages")
    provider: str = Field("deepseek", description="Model provider")
    model: Optional[str] = Field(None, description="Model name")

@app.post("/chat/direct/stream")
async def chat_direct_stream(request: DirectChatRequest):
    """
    Direct LLM streaming endpoint without RAG.
    Used for wiki structure determination where the file tree is already in the prompt.
    """
    import asyncio

    try:
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1]
        prompt = last_message.get("content", "")

        model_config = get_model_config_func(request.provider, request.model)["model_kwargs"]

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
            model = OpenAIClient(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
        else:
            # DeepSeek or any OpenAI-compatible provider
            model = OpenAIClient()
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
                                    yield text
            except Exception as e:
                logger.error(f"Error in direct LLM streaming: {str(e)}")
                yield f"\nError: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in chat_direct_stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
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
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")


        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
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
