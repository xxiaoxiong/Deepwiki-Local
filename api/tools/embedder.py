import adalflow as adal

from api.config import configs, get_embedder_type


def get_embedder(is_local_ollama: bool = False, use_google_embedder: bool = False, embedder_type: str = None) -> adal.Embedder:
    """根据配置或参数获取嵌入器实例。

    Args:
        is_local_ollama: 旧版参数，用于指定 Ollama 嵌入器（已废弃，建议使用 embedder_type）
        use_google_embedder: 旧版参数，用于指定 Google 嵌入器（已废弃，建议使用 embedder_type）
        embedder_type: 直接指定嵌入器类型（'ollama' 或 'openai'）

    Returns:
        adal.Embedder: 已配置好的嵌入器实例
    """
    # 根据参数或自动检测确定要使用的嵌入器配置
    if embedder_type:
        # 优先使用显式指定的嵌入器类型
        if embedder_type == 'ollama':
            embedder_config = configs["embedder_ollama"]
        else:  # 默认使用 openai 兼容嵌入器
            embedder_config = configs["embedder"]
    elif is_local_ollama:
        # 旧版兼容：通过 is_local_ollama 参数指定使用 Ollama
        embedder_config = configs["embedder_ollama"]
    else:
        # 自动检测：根据当前配置决定使用哪种嵌入器
        current_type = get_embedder_type()
        if current_type == 'ollama':
            embedder_config = configs["embedder_ollama"]
        else:
            embedder_config = configs["embedder"]

    # --- 初始化嵌入器模型客户端 ---
    model_client_class = embedder_config["model_client"]
    # 如果配置中有初始化参数，则传入；否则无参初始化
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    # 构造嵌入器所需的基本参数
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}

    # 创建嵌入器实例
    embedder = adal.Embedder(**embedder_kwargs)

    # 若配置中有 batch_size，将其作为属性设置（非构造参数）
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
