"""商品检索工具:根据用户问题匹配最相关的商品及其客观信息。"""
from __future__ import annotations
from typing import Any

from .base import Tool
from ...data.products import all_products
from ...knowledge import embedding


class ProductSearchTool(Tool):
    name = "product_search"
    description = "检索店铺商品目录,返回最相关商品的客观信息(名称、价格、库存、规格、卖点)。"

    def run(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        qvec = embedding.embed(query)
        best = None
        best_score = 0.0
        for p in all_products():
            text = " ".join([p["name"], p["category"], p["specs"],
                             p["highlight"], " ".join(p.get("keywords", []))])
            score = embedding.cosine(qvec, embedding.embed(text))
            if score > best_score:
                best_score, best = score, p
        if best and best_score >= 0.06:
            return {"found": True, "score": round(best_score, 3),
                    "product": {k: best[k] for k in ("id", "name", "category", "price",
                                                     "stock", "specs", "highlight")}}
        return {"found": False, "score": round(best_score, 3),
                "products": [{"name": p["name"], "price": p["price"],
                              "category": p["category"]} for p in all_products()]}
