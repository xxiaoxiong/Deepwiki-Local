import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

logger = logging.getLogger(__name__)

from api.openai_client import OpenAIClient
from adalflow.components.model_client.ollama_client import OllamaClient

# 从环境变量获取 OpenAI 相关配置
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.deepseek.com/v1')

# vLLM 配置（用于内网部署）
VLLM_API_KEY = os.environ.get('VLLM_API_KEY', 'not-needed')
VLLM_BASE_URL = os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')

# 运行时提供商 URL 覆盖（通过 UI 设置，重启后不保留）
runtime_overrides: Dict[str, Any] = {
    "provider_urls": {},  # 示例: {"vllm": "http://10.0.0.1:8000/v1", "deepseek": "https://..."}
}


def get_provider_base_url(provider: str) -> str:
    """获取指定提供商的有效 base URL，优先使用运行时覆盖值。

    Args:
        provider: 提供商标识符，如 'vllm'、'deepseek'、'ollama'

    Returns:
        str: 该提供商的 base URL
    """
    # 优先检查运行时覆盖
    override_url = runtime_overrides["provider_urls"].get(provider)
    if override_url:
        return override_url
    # 回退到环境变量/默认值
    if provider == "vllm":
        return VLLM_BASE_URL
    elif provider == "deepseek":
        return OPENAI_BASE_URL or 'https://api.deepseek.com/v1'
    elif provider == "ollama":
        return os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    else:
        return OPENAI_BASE_URL or 'https://api.openai.com/v1'


def set_provider_base_url(provider: str, url: str):
    """在运行时覆盖指定提供商的 base URL（重启后失效）。

    Args:
        provider: 提供商标识符
        url: 新的 base URL
    """
    runtime_overrides["provider_urls"][provider] = url
    logger.info(f"已为提供商 '{provider}' 设置运行时 base URL 覆盖: {url}")


# 将 API 密钥写入环境变量（供项目其他模块使用）
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

# 旧版兼容别名（保持向后兼容）
OPENROUTER_API_KEY = None
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_SESSION_TOKEN = None
AWS_REGION = None
AWS_ROLE_ARN = None

# Wiki 认证配置
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']  # 是否开启认证模式
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')      # 认证码

# 嵌入器类型配置
EMBEDDER_TYPE = os.environ.get('DEEPWIKI_EMBEDDER_TYPE', 'openai').lower()

# 配置文件目录（可通过环境变量自定义）
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)

# 客户端类名到类对象的映射
CLIENT_CLASSES = {
    "OpenAIClient": OpenAIClient,
    "OllamaClient": OllamaClient,
}


def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """递归替换配置中的环境变量占位符（如 "${ENV_VAR}"）。

    遍历嵌套的字典、列表和字符串，将 ${VAR_NAME} 格式的占位符
    替换为对应的环境变量值。若环境变量不存在，保留原占位符并输出警告。

    Args:
        config: 待处理的配置对象（字典、列表、字符串或其他类型）

    Returns:
        替换占位符后的配置对象
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"环境变量占位符 '{original_placeholder}' 未在环境中找到，将保留原占位符。"
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        # 数字、布尔值、None 等类型直接返回
        return config


def load_json_config(filename):
    """加载 JSON 格式的配置文件，并自动替换环境变量占位符。

    Args:
        filename: 配置文件名（相对于配置目录）

    Returns:
        dict: 解析并处理后的配置字典，失败时返回空字典
    """
    try:
        # 优先使用环境变量指定的配置目录
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            config_path = Path(__file__).parent / "config" / filename

        logger.info(f"正在从 {config_path} 加载配置")

        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)  # 替换环境变量占位符
            return config
    except Exception as e:
        logger.error(f"加载配置文件 {filename} 时出错: {str(e)}")
        return {}


def load_generator_config():
    """加载生成器（LLM）模型配置，并为每个提供商绑定对应的客户端类。

    Returns:
        dict: 包含提供商和模型配置的字典
    """
    generator_config = load_json_config("generator.json")

    # 为每个提供商绑定客户端类
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            # 优先通过 client_class 字段查找客户端类
            if provider_config.get("client_class") in CLIENT_CLASSES:
                provider_config["model_client"] = CLIENT_CLASSES[provider_config["client_class"]]
            # 回退到基于 provider_id 的默认映射
            elif provider_id in ["deepseek", "vllm", "ollama"]:
                default_map = {
                    "deepseek": OpenAIClient,
                    "vllm": OpenAIClient,
                    "ollama": OllamaClient,
                }
                provider_config["model_client"] = default_map[provider_id]
            else:
                logger.warning(f"未知提供商或客户端类: {provider_id}")

    return generator_config


def load_embedder_config():
    """加载嵌入器配置，并为每个嵌入器绑定对应的客户端类。

    Returns:
        dict: 包含嵌入器配置的字典
    """
    embedder_config = load_json_config("embedder.json")

    # 处理各嵌入器的客户端类绑定
    for key in ["embedder", "embedder_ollama"]:
        if key in embedder_config and "client_class" in embedder_config[key]:
            class_name = embedder_config[key]["client_class"]
            if class_name in CLIENT_CLASSES:
                embedder_config[key]["model_client"] = CLIENT_CLASSES[class_name]

    return embedder_config


def get_embedder_config():
    """根据 DEEPWIKI_EMBEDDER_TYPE 获取当前嵌入器配置。

    Returns:
        dict: 包含已解析 model_client 的嵌入器配置字典
    """
    embedder_type = EMBEDDER_TYPE
    if embedder_type == 'ollama' and 'embedder_ollama' in configs:
        return configs.get("embedder_ollama", {})
    else:
        return configs.get("embedder", {})


def is_ollama_embedder():
    """检查当前嵌入器是否使用 OllamaClient。

    Returns:
        bool: 使用 OllamaClient 时返回 True，否则返回 False
    """
    embedder_config = get_embedder_config()
    if not embedder_config:
        return False

    # 检查 model_client 是否为 OllamaClient 类
    model_client = embedder_config.get("model_client")
    if model_client:
        return model_client.__name__ == "OllamaClient"

    # 回退：检查 client_class 字符串
    client_class = embedder_config.get("client_class", "")
    return client_class == "OllamaClient"


def get_embedder_type():
    """获取当前嵌入器类型。

    Returns:
        str: 'ollama' 或 'openai'（默认）
    """
    if is_ollama_embedder():
        return 'ollama'
    else:
        return 'openai'


def load_repo_config():
    """加载仓库和文件过滤器配置。

    Returns:
        dict: 仓库配置字典
    """
    return load_json_config("repo.json")


def load_lang_config():
    """加载语言配置，若配置文件缺失或格式错误则使用默认配置。

    Returns:
        dict: 包含 supported_languages 和 default 字段的语言配置字典
    """
    default_config = {
        "supported_languages": {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)",
            "pt-br": "Brazilian Portuguese (Português Brasileiro)",
            "fr": "Français (French)",
            "ru": "Русский (Russian)"
        },
        "default": "en"
    }

    loaded_config = load_json_config("lang.json")

    if not loaded_config:
        return default_config

    if "supported_languages" not in loaded_config or "default" not in loaded_config:
        logger.warning("语言配置文件 'lang.json' 格式错误，将使用默认语言配置。")
        return default_config

    return loaded_config


# 默认排除的目录列表
DEFAULT_EXCLUDED_DIRS: List[str] = [
    # 虚拟环境和包管理器
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # 版本控制
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # 缓存和编译文件
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # 构建和发布目录
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # 文档目录
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE 特定目录
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # 日志和临时文件
    "./logs/", "./log/", "./tmp/", "./temp/",
]

# 默认排除的文件列表
DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# 初始化空配置字典
configs = {}

# 加载所有配置文件
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
lang_config = load_lang_config()

# 更新生成器配置
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "deepseek")
    configs["providers"] = generator_config.get("providers", {})

# 更新嵌入器配置
if embedder_config:
    for key in ["embedder", "embedder_ollama", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# 更新仓库配置
if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# 更新语言配置
if lang_config:
    configs["lang_config"] = lang_config


def get_model_config(provider="deepseek", model=None):
    """获取指定提供商和模型的配置。

    Args:
        provider (str): 模型提供商（'deepseek'、'vllm'、'ollama' 等）
        model (str): 模型名称，为 None 时使用提供商的默认模型

    Returns:
        dict: 包含 model_client 和 model_kwargs 的配置字典

    Raises:
        ValueError: 提供商或模型配置不存在时抛出
    """
    # 检查提供商配置是否已加载
    if "providers" not in configs:
        raise ValueError("提供商配置未加载")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"未找到提供商 '{provider}' 的配置")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"提供商 '{provider}' 未指定模型客户端")

    # 未指定模型时使用提供商默认模型
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"提供商 '{provider}' 未指定默认模型")

    # 获取模型参数（若存在）
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        # 模型不在列表中时回退到默认模型的参数
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"][default_model]

    result = {
        "model_client": model_client,
    }

    # 根据提供商类型构造 model_kwargs
    if provider == "ollama":
        # Ollama 使用 options 嵌套参数结构
        if "options" in model_params:
            result["model_kwargs"] = {"model": model, **model_params["options"]}
        else:
            result["model_kwargs"] = {"model": model}
    else:
        # 其他提供商使用扁平参数结构
        result["model_kwargs"] = {"model": model, **model_params}

    return result
 