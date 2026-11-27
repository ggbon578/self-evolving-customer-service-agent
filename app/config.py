"""全局配置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data.db"
STATIC_DIR = BASE_DIR / "static"

# 从 .env 读取密钥等敏感配置(文件不存在时静默跳过)
load_dotenv(BASE_DIR / ".env")

# Agent Loop 边界控制
MAX_LOOP_STEPS = 6          # 单次对话内最多循环步数,防死循环
LOOP_TIMEOUT_SECONDS = 30   # 单次对话超时

# 会话历史滑动窗口:每个会话只保留最近 N 条消息(用户+最终回复)进入下一轮上下文,
# 超出丢最旧的——防止长对话历史无限膨胀、稀释模型注意力。
SESSION_MAX_MESSAGES = int(os.getenv("SESSION_MAX_MESSAGES", "6"))

# 知识库检索:命中阈值(余弦相似度),低于此值视为"未命中"
KB_HIT_THRESHOLD = 0.45
KB_TOP_K = 3

# 套路库(playbook)语义检索:每次只召回最相关的 top-k 条样本,避免全量塞爆上下文
PLAYBOOK_TOP_K = int(os.getenv("PLAYBOOK_TOP_K", "5"))

# 召回质量门槛:相似度低于该值的样本视为无关,不返回。
# 修复"0 分也被硬凑进 top-k 喂给模型"的问题——宁缺毋滥,没有匹配就坦诚返回空。
PLAYBOOK_MIN_SCORE = float(os.getenv("PLAYBOOK_MIN_SCORE", "0.15"))

# 每轮注入系统提示的示范条数(取召回结果中最相关的前 N 条),
# 控制上下文长度,避免"5 条全量注入 → 上千字 → 注意力稀释"。
PLAYBOOK_INJECT_TOP_N = int(os.getenv("PLAYBOOK_INJECT_TOP_N", "2"))

# 注入话术的最大字符数,超出截断(策略 note 保留全文——它是应对方式的精华,
# 砍策略会让模型只会盲抄话术;真正占字数的话术才是砍的大头)
PLAYBOOK_ANSWER_MAX_CHARS = int(os.getenv("PLAYBOOK_ANSWER_MAX_CHARS", "80"))

# 注入策略(note)的最大字符数:策略保留但也不许超长,超出截断
PLAYBOOK_NOTE_MAX_CHARS = int(os.getenv("PLAYBOOK_NOTE_MAX_CHARS", "60"))

# 选用的 LLM 提供方:mock(离线桩)| ark(真实豆包/火山引擎 ARK)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")

# 豆包(火山引擎 ARK)配置,OpenAI 兼容接口
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_CHAT_MODEL = os.getenv("ARK_CHAT_MODEL", "")
# 豆包 embedding 模型(multimodal 接口),用于套路库语义检索
ARK_EMBED_MODEL = os.getenv("ARK_EMBED_MODEL", "doubao-embedding-vision-251215")
