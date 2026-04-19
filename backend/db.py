from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend import config


class Base(DeclarativeBase):
    """项目所有 SQLAlchemy 模型的基类。"""


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def build_mysql_url() -> str:
    # 密码里可能包含特殊字符，先转义再拼接连接串。
    password = quote_plus(config.MYSQL_PASSWORD or "")
    return (
        f"mysql+pymysql://{config.MYSQL_USER}:{password}"
        f"@{config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}"
        "?charset=utf8mb4"
    )


def init_mysql() -> None:
    global _engine, _session_factory

    # 启动阶段只初始化一次，避免重复创建连接池。
    if _engine is not None and _session_factory is not None:
        return

    _engine = create_engine(
        build_mysql_url(),
        # 连接复用前先探活，避免使用到已失效连接。
        pool_pre_ping=True,
        future=True,
    )
    # 关闭自动过期，便于 service 提交后直接读取对象字段返回响应。
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("MySQL 引擎尚未初始化")
    return _engine


def get_session_factory() -> sessionmaker:
    if _session_factory is None:
        raise RuntimeError("MySQL 会话工厂尚未初始化")
    return _session_factory
