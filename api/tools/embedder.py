import adalflow as adal

from api.config import configs, get_embedder_type


def get_embedder(is_local_ollama: bool = False, use_google_embedder: bool = False, embedder_type: str = None):
    """根据配置或参数获取嵌入器实例。

    Args:
        is_local_ollama: 旧版参数，用于指定 Ollama 嵌入器（已废弃，建议使用 embedder_type）
        use_google_embedder: 旧版参数，用于指定 Google 嵌入器（已废弃，建议使用 embedder_type）
        embedder_type: 直接指定嵌入器类型（'local'、'ollama' 或 'openai'）

    Returns:
        嵌入器实例（SentenceTransformerEmbedder / adal.Embedder）
    """
    # 确定实际嵌入器类型
    resolved_type = embedder_type
    if resolved_type is None:
        if is_local_ollama:
            resolved_type = 'ollama'
        else:
            resolved_type = get_embedder_type()

    # --- 本地 sentence-transformers 嵌入器 ---
    if resolved_type == 'local':
        import os
        from api.sentence_transformer_client import SentenceTransformerEmbedder
        local_cfg = configs.get("embedder_local", {})
        # 优先使用 LOCAL_EMBEDDING_MODEL 环境变量（docker-compose.yml 中设置），其次读取配置文件
        model_name = os.environ.get("LOCAL_EMBEDDING_MODEL") or local_cfg.get("model_name", "BAAI/bge-small-zh-v1.5")
        model_path = os.environ.get("LOCAL_EMBEDDING_MODEL_PATH") or local_cfg.get("model_path", None)
        # 若 model_path 含未解析的占位符则忽略
        if model_path and "${" in str(model_path):
            model_path = None
        return SentenceTransformerEmbedder(model_name=model_name, model_path=model_path)

    # --- 根据类型选择嵌入器配置 ---
    if resolved_type == 'ollama':
        embedder_config = configs["embedder_ollama"]
    else:
        embedder_config = configs["embedder"]

    # --- 初始化嵌入器模型客户端 ---
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}
    embedder = adal.Embedder(**embedder_kwargs)

    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
