"""套路召回工具:按客户当前问句做 embedding 语义检索,取回最相关的示范问答与套路,
供大模型模仿作答。

这是本系统的核心:回答任何咨询前先调它。只召回与本次问句最相关的
top-N 条(带相似度门槛过滤),并对话术做截断——库变大也不会撑爆上下文。
"""
from __future__ import annotations
from typing import Any

from .base import Tool
from ...services import playbook_service
from ... import config


class RecallPlaybookTool(Tool):
    name = "recall_playbook"
    description = ("检索店长教过的、与客户当前问句最相关的示范问答与套路(客户问题+参考话术+"
                   "这么答的套路)。回答任何客户咨询前必须先调用它,然后用语义理解判断客户这句话"
                   "命中哪条示范的套路,严格模仿该套路的策略与话术风格来回答。")

    def run(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        # 只取最相关的 top-N 条并裁剪话术(与注入提示词的条数对齐):
        # 之前返回 top-5 全文,一条两三百字、5 条上千字,是上下文过载的主因之一。
        samples = playbook_service.recall_relevant(query)
        return {"count": len(samples), "samples": samples}
