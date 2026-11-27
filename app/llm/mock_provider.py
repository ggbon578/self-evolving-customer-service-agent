"""规则桩 LLM(离线演示用)。

这是为了**离线、可重复**地演示电商客服场景下的 Agent Loop 而写的"假大脑":
它读对话历史(含已执行的工具结果),用规则模拟真实模型的"决定调哪个工具 / 何时收尾"。

>>> 换成真实模型(ArkProvider)时,语义理解取代这里的规则即可,Loop 与工具都不用动。
"""
from __future__ import annotations
import re
from typing import Any

from .base import LLMProvider, LLMDecision, ToolCall

# 各类意图的关键词
_HANDOFF_RE = re.compile(r"(转人工|人工客服|真人|人工)")
_FACT_RE = re.compile(r"(价格|多少钱|贵|便宜|库存|有货|规格|参数|发货|多久|续航|防水|降噪|适合)")
_REVIEW_RE = re.compile(r"(评价|买家|口碑|质量|好用|谁买过|反馈|怎么样)")


def _call(name: str, tool_input: dict[str, Any], thought: str) -> LLMDecision:
    """构造单工具调用决策(桩场景一步只调一个工具)。"""
    return LLMDecision(
        type="tool_call",
        tool_calls=[ToolCall(id=f"call_{name}", name=name, input=tool_input)],
        thought=thought,
    )


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def _tool_results_since_last_user(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """返回 {tool_name: result} —— 本轮用户提问后已执行过的工具及其结果。"""
    results: dict[str, Any] = {}
    collecting = False
    for m in messages:
        if m.get("role") == "user":
            results = {}          # 遇到新的用户提问,重置
            collecting = True
            continue
        if collecting and m.get("role") == "tool":
            results[m.get("name")] = m.get("result")
    return results


class MockLLMProvider(LLMProvider):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMDecision:
        question = _last_user_message(messages)
        done = _tool_results_since_last_user(messages)

        # 0) 用户明确要求转人工
        if _HANDOFF_RE.search(question):
            if "handoff" not in done:
                return _call("handoff", {"question": question, "context": "用户主动要求转人工"},
                             "用户明确要求转人工,直接转接。")
            return LLMDecision(type="final", thought="已转接。",
                               content="已为您转接人工客服,请稍候,客服会尽快跟进。")

        # 1) 回答任何咨询前,先召回店长教过的示范问答与套路
        if "recall_playbook" not in done:
            return _call("recall_playbook", {"query": question},
                         "先召回店长教过的示范问答与套路。")

        pb = done.get("recall_playbook") or {}

        # 1.1 套路库为空 —— 还没教过任何东西,转人工
        if not pb.get("count"):
            if "handoff" not in done:
                return _call("handoff",
                             {"question": question, "context": "套路库为空,尚未教过任何内容。"},
                             "套路库还是空的,我没有把握,转人工。")
            return LLMDecision(type="final", thought="已转人工。",
                               content="这个问题我还没学过,已为您转接人工客服处理。")

        # 2) 涉及商品客观事实(价格/库存/参数等)—— 查商品库
        if _FACT_RE.search(question) and "product_search" not in done:
            return _call("product_search", {"query": question},
                         "问题涉及商品客观信息,检索商品库核对事实。")

        # 3) 需要用真实买家反馈打消顾虑 —— 查买家评价
        if _REVIEW_RE.search(question) and "buyer_reviews" not in done:
            return _call("buyer_reviews", {"query": question},
                         "需要真实买家评价作答,检索评价库。")

        # 4) 收尾:按召回到的套路样本 + 已查到的信息作答
        return LLMDecision(type="final", thought="按召回到的套路作答。",
                           content=_answer(pb, done))


def _answer(pb: dict[str, Any], done: dict[str, Any]) -> str:
    """把套路样本 + 商品信息 + 买家评价拼成回复(真实模型会语义精判并自然生成)。"""
    samples = pb.get("samples") or []
    product = (done.get("product_search") or {}).get("product")
    reviews = (done.get("buyer_reviews") or {}).get("reviews")

    parts: list[str] = []
    if samples:
        parts.append(samples[0].get("answer", ""))
    if product:
        parts.append(f"(参考:{product['name']},{product.get('price', '')},{product.get('specs', '')})")
    if reviews:
        r = reviews[0]
        parts.append(f"比如{r['buyer']}({r['city']})的评价——{r['comment']}。")
    return "".join(p for p in parts if p) or "您好,请问您具体想了解哪方面呢?"
