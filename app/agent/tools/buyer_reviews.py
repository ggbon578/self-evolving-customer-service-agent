"""买家评价查询工具:按商品/关注点检索真实买家反馈(mock 数据)。

当套路要求"引用买家反馈打消顾虑"时,由 Agent 调用它拿到具体评价,
避免笼统地说"质量很好、很多人买"。
"""
from __future__ import annotations
from typing import Any

from .base import Tool
from ...data.reviews import all_reviews


class BuyerReviewsTool(Tool):
    name = "buyer_reviews"
    description = ("按商品名称或客户关注点(质量/续航/噪音/防水等)检索真实买家评价,返回具体评价"
                   "(买家、城市、购买商品、评价内容)。当需要用真实反馈给客户信心时调用。")

    def run(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        q = (query or "").lower()
        hits = [r for r in all_reviews()
                if q in r["product"].lower()
                or any(kw in q or q in kw for kw in r.get("keywords", []))
                or any(kw in q for kw in r.get("keywords", []))]
        if not hits:
            hits = all_reviews()[:2]
        return {"count": len(hits), "reviews": hits}
