"""业务设定:店长可在后台看/改的『全局人设 + 语气 + 铁律』,会注入系统提示词。

这是与"逐条教学样本"不同的一层——承载全局策略(简洁、绝不直接报价等),
这类硬规则不适合靠逐条样本归纳,由店长直接维护更可靠。
"""
from __future__ import annotations

from ..db import cursor

_KEY = "business_directive"
_SUMMARY_KEY = "playbook_summary"


def _get(key: str, default: str = "") -> str:
    with cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
    return row["value"] if row else default


def _set(key: str, value: str) -> None:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

DEFAULT_DIRECTIVE = (
    "【人设】你是『星辰数码』官方旗舰店的客服,回复简洁口语化,一次一般不超过 2-3 句;"
    "先回应重点,别长篇大论、别堆砌。\n"
    "【铁律】\n"
    "1. 商品的价格、库存、参数等客观信息必须来自 product_search 或店长教过的样本,绝不编造。\n"
    "2. 客户比价、犹豫时:先挖掘真实需求(用途/场景),再用商品卖点和买家评价塑造价值,最后自然引导下单。\n"
    "3. 不夸大参数、不承诺虚假功效;售后问题(七天无理由退换、一年质保、运费险)按店铺政策回答,拿不准就转人工。\n"
    "4. 任何时候遵循店长教过的套路,客户信息不足时先追问(用途、预算、型号偏好),再推进。"
)


def get_directive() -> str:
    return _get(_KEY, DEFAULT_DIRECTIVE)


def set_directive(text: str) -> dict:
    _set(_KEY, text)
    return {"ok": True}


def get_summary() -> str:
    """已保存的套路总纲(可能被店长手动改写过);空串表示还没生成过。"""
    return _get(_SUMMARY_KEY, "")


def set_summary(text: str) -> dict:
    _set(_SUMMARY_KEY, text)
    return {"ok": True}
