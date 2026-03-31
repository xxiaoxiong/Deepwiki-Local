"""
定时拉取远程仓库并重新分析的调度模块。
使用 APScheduler 实现定时任务，配置以 JSON 文件持久化。
"""
import json
import os
import logging
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# 调度配置文件路径
SCHEDULE_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".adalflow", "schedules")
SCHEDULE_CONFIG_FILE = os.path.join(SCHEDULE_CONFIG_DIR, "schedules.json")

_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()


@dataclass
class ScheduleConfig:
    """单个仓库的定时拉取配置。"""
    id: str
    name: str                        # 任务名称（用于显示）
    repo_url: str                    # 仓库 URL
    repo_type: str = "gitlab"        # github / gitlab / gitee / gitea
    access_token: str = ""           # 访问令牌（可选）
    interval_hours: int = 24         # 拉取间隔（小时）
    cron_expr: str = ""              # Cron 表达式（优先于 interval_hours）
    enabled: bool = True             # 是否启用
    excluded_dirs: str = ""          # 排除目录（逗号分隔）
    excluded_files: str = ""         # 排除文件（逗号分隔）
    last_run: str = ""               # 上次运行时间
    last_status: str = ""            # 上次运行状态
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _load_schedules() -> List[ScheduleConfig]:
    """从 JSON 文件加载调度配置列表。"""
    os.makedirs(SCHEDULE_CONFIG_DIR, exist_ok=True)
    if not os.path.exists(SCHEDULE_CONFIG_FILE):
        return []
    try:
        with open(SCHEDULE_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ScheduleConfig(**item) for item in data]
    except Exception as e:
        logger.error(f"加载调度配置失败: {e}")
        return []


def _save_schedules(schedules: List[ScheduleConfig]):
    """将调度配置列表保存到 JSON 文件。"""
    os.makedirs(SCHEDULE_CONFIG_DIR, exist_ok=True)
    try:
        with open(SCHEDULE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in schedules], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存调度配置失败: {e}")


def _run_pull_and_analyze(schedule_id: str):
    """
    执行定时拉取和重新分析任务。
    先删除旧的 embeddings 缓存，再重新分析。
    """
    schedules = _load_schedules()
    cfg = next((s for s in schedules if s.id == schedule_id), None)
    if cfg is None:
        logger.error(f"找不到调度任务: {schedule_id}")
        return
    if not cfg.enabled:
        logger.info(f"调度任务 {schedule_id} 已禁用，跳过")
        return

    logger.info(f"[定时任务] 开始拉取仓库: {cfg.repo_url}")

    # 更新运行时间
    cfg.last_run = datetime.now().isoformat()
    cfg.last_status = "running"
    _update_schedule_status(schedule_id, last_run=cfg.last_run, last_status="running")

    try:
        from api.data_pipeline import DatabaseManager
        from api.config import get_embedder_type

        excluded_dirs = [d.strip() for d in cfg.excluded_dirs.split(",") if d.strip()] if cfg.excluded_dirs else None
        excluded_files = [f.strip() for f in cfg.excluded_files.split(",") if f.strip()] if cfg.excluded_files else None

        db_manager = DatabaseManager()
        embedder_type = get_embedder_type()
        db_manager.prepare_database(
            repo_url_or_path=cfg.repo_url,
            repo_type=cfg.repo_type,
            access_token=cfg.access_token or None,
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
        )
        logger.info(f"[定时任务] 仓库分析完成: {cfg.repo_url}")
        _update_schedule_status(schedule_id, last_status="success")
    except Exception as e:
        logger.error(f"[定时任务] 仓库分析失败 {cfg.repo_url}: {e}")
        _update_schedule_status(schedule_id, last_status=f"error: {str(e)[:200]}")


def _update_schedule_status(schedule_id: str, **kwargs):
    """更新调度任务的状态字段。"""
    schedules = _load_schedules()
    for s in schedules:
        if s.id == schedule_id:
            for k, v in kwargs.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            break
    _save_schedules(schedules)


def get_scheduler() -> BackgroundScheduler:
    """获取全局调度器实例（懒初始化）。"""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
            _scheduler.start()
            logger.info("APScheduler 已启动")
            _reload_all_jobs()
    return _scheduler


def _reload_all_jobs():
    """从配置文件重新加载所有调度任务到调度器中。"""
    if _scheduler is None:
        return
    _scheduler.remove_all_jobs()
    schedules = _load_schedules()
    for cfg in schedules:
        if cfg.enabled:
            _add_job_to_scheduler(cfg)
    logger.info(f"已加载 {len([s for s in schedules if s.enabled])} 个定时任务")


def _add_job_to_scheduler(cfg: ScheduleConfig):
    """将单个调度配置注册到 APScheduler。"""
    if _scheduler is None:
        return
    try:
        if cfg.cron_expr:
            # 使用 cron 表达式
            parts = cfg.cron_expr.strip().split()
            if len(parts) == 5:
                trigger = CronTrigger(
                    minute=parts[0], hour=parts[1], day=parts[2],
                    month=parts[3], day_of_week=parts[4]
                )
            else:
                logger.warning(f"无效的 cron 表达式: {cfg.cron_expr}，改用 interval")
                trigger = IntervalTrigger(hours=cfg.interval_hours or 24)
        else:
            trigger = IntervalTrigger(hours=cfg.interval_hours or 24)

        _scheduler.add_job(
            _run_pull_and_analyze,
            trigger=trigger,
            args=[cfg.id],
            id=cfg.id,
            name=cfg.name,
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info(f"已注册定时任务: {cfg.name} ({cfg.id})")
    except Exception as e:
        logger.error(f"注册定时任务失败 {cfg.id}: {e}")


# ---- CRUD 接口 ----

def list_schedules() -> List[Dict[str, Any]]:
    """列出所有调度配置。"""
    schedules = _load_schedules()
    result = []
    for s in schedules:
        d = asdict(s)
        # 获取下次运行时间
        scheduler = get_scheduler()
        job = scheduler.get_job(s.id)
        d["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else ""
        result.append(d)
    return result


def get_schedule(schedule_id: str) -> Optional[Dict[str, Any]]:
    """获取单个调度配置。"""
    schedules = _load_schedules()
    for s in schedules:
        if s.id == schedule_id:
            d = asdict(s)
            scheduler = get_scheduler()
            job = scheduler.get_job(s.id)
            d["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else ""
            return d
    return None


def create_schedule(data: Dict[str, Any]) -> Dict[str, Any]:
    """创建新的调度配置。"""
    import uuid
    schedules = _load_schedules()
    cfg = ScheduleConfig(
        id=str(uuid.uuid4()),
        name=data.get("name", "Unnamed"),
        repo_url=data.get("repo_url", ""),
        repo_type=data.get("repo_type", "gitlab"),
        access_token=data.get("access_token", ""),
        interval_hours=int(data.get("interval_hours", 24)),
        cron_expr=data.get("cron_expr", ""),
        enabled=bool(data.get("enabled", True)),
        excluded_dirs=data.get("excluded_dirs", ""),
        excluded_files=data.get("excluded_files", ""),
    )
    schedules.append(cfg)
    _save_schedules(schedules)
    if cfg.enabled:
        _add_job_to_scheduler(cfg)
    return asdict(cfg)


def update_schedule(schedule_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """更新调度配置。"""
    schedules = _load_schedules()
    for i, s in enumerate(schedules):
        if s.id == schedule_id:
            updatable = ["name", "repo_url", "repo_type", "access_token",
                         "interval_hours", "cron_expr", "enabled",
                         "excluded_dirs", "excluded_files"]
            for key in updatable:
                if key in data:
                    val = data[key]
                    if key == "interval_hours":
                        val = int(val)
                    elif key == "enabled":
                        val = bool(val)
                    setattr(s, key, val)
            schedules[i] = s
            _save_schedules(schedules)
            # 重新注册 job
            scheduler = get_scheduler()
            scheduler.remove_job(schedule_id) if scheduler.get_job(schedule_id) else None
            if s.enabled:
                _add_job_to_scheduler(s)
            return asdict(s)
    return None


def delete_schedule(schedule_id: str) -> bool:
    """删除调度配置。"""
    schedules = _load_schedules()
    new_schedules = [s for s in schedules if s.id != schedule_id]
    if len(new_schedules) == len(schedules):
        return False
    _save_schedules(new_schedules)
    scheduler = get_scheduler()
    if scheduler.get_job(schedule_id):
        scheduler.remove_job(schedule_id)
    return True


def trigger_schedule_now(schedule_id: str):
    """立即触发一次调度任务（不等待下次定时）。"""
    thread = threading.Thread(
        target=_run_pull_and_analyze,
        args=(schedule_id,),
        daemon=True
    )
    thread.start()
    logger.info(f"已手动触发调度任务: {schedule_id}")


def shutdown_scheduler():
    """关闭调度器（应用退出时调用）。"""
    global _scheduler
    with _scheduler_lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            logger.info("APScheduler 已关闭")
