import os
import sys
import logging
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

from api.logging_config import setup_logging

# 配置应用日志
setup_logging()
logger = logging.getLogger(__name__)

# 配置 watchfiles 日志，以便在文件变更时显示路径信息
watchfiles_logger = logging.getLogger("watchfiles.main")
watchfiles_logger.setLevel(logging.DEBUG)  # 开启 DEBUG 级别以查看文件路径

# 将当前目录加入 Python 路径，以便正确导入 api 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在导入 uvicorn 之前应用 watchfiles 猴子补丁（仅开发模式下）
is_development = os.environ.get("NODE_ENV") != "production"
if is_development:
    import watchfiles
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "logs")

    original_watch = watchfiles.watch
    def patched_watch(*args, **kwargs):
        # 只监视 api 目录中的 Python 文件和子目录，排除 logs 子目录
        # 避免日志文件变更触发不必要的热重载
        api_subdirs = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and item != "logs":
                api_subdirs.append(item_path)
            elif os.path.isfile(item_path) and item.endswith(".py"):
                api_subdirs.append(item_path)

        return original_watch(*api_subdirs, **kwargs)
    watchfiles.watch = patched_watch

import uvicorn

# 检查必要的环境变量是否已设置
required_env_vars = ['OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"缺少环境变量: {', '.join(missing_vars)}")
    logger.warning("缺少这些变量可能导致部分功能无法正常使用。")

if __name__ == "__main__":
    # 从环境变量获取端口号，默认使用 8001
    port = int(os.environ.get("PORT", 8001))

    # 在此处导入 app，确保环境变量在导入前已设置完毕
    from api.api import app

    logger.info(f"正在启动 Streaming API，端口: {port}")

    # 使用 uvicorn 运行 FastAPI 应用
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,  # 开发模式下开启热重载
        reload_excludes=["**/logs/*", "**/__pycache__/*", "**/*.pyc"] if is_development else None,  # 排除不需要监视的目录
    )
