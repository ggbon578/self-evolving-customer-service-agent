"""对话服务:维护会话上下文、驱动 Agent Loop、记录指标。"""
from __future__ import annotations
from typing import Any

from ..agent.loop import AgentLoop, LoopResult
from ..llm.mock_provider import MockLLMProvider
from ..llm.base import LLMProvider
from ..db import cursor
from .. import config

# 简单的进程内会话存储(演示用;生产可换 Redis/DB)
_SESSIONS: dict[str, list[dict[str, Any]]] = {}


def _build_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "ark":
        from ..llm.ark_provider import ArkLLMProvider
        return ArkLLMProvider()
    return MockLLMProvider()


_LOOP = AgentLoop(_build_provider())


# 验收专用的总纲覆盖:影子跑分时要看"这批候选样本重新归纳后的总纲"会带来什么后果,
# 但不能真把没验证的总纲写进 settings。用进程内变量临时顶替,跑完置空。
_SUMMARY_OVERRIDE: str | None = None


def set_summary_override(text: str | None) -> None:
    global _SUMMARY_OVERRIDE
    _SUMMARY_OVERRIDE = text


def effective_system_prompt() -> str:
    """完整生效的系统提示词(技术流程 + 业务设定 + 套路总纲),供后台查看与聊天使用。"""
    from . import settings_service
    parts = [_BASE_PROMPT]
    parts.append(
        "\n\n【业务设定(店长在后台可改,始终优先遵守,为最高铁律)】\n"
        + settings_service.get_directive()
    )
    summary = (_SUMMARY_OVERRIDE if _SUMMARY_OVERRIDE is not None
               else settings_service.get_summary()).strip()
    if summary:
        parts.append(
            "\n\n【套路总纲(店长已确认的应答策略,按此执行;"
            "recall_playbook 取到的样本仅作具体话术参照,与本总纲冲突时以本总纲为准)】\n"
            + summary
        )
    return "".join(parts)


_BASE_PROMPT = (
    "你是『星辰数码』官方旗舰店的电商客服。你的回答必须遵循店长教过的『套路』,而不是有问必直答"
    "(例如客户问价,套路可能是先确认需求、结合卖点与优惠引导下单)。\n\n"
    "可用工具:\n"
    "- recall_playbook:取回店长教过的示范问答与套路(客户问题+标准答案+这么答的原因)。"
    "注意:系统每轮已自动召回并把结果放进系统提示,通常无需再调用它。\n"
    "- product_search:检索店铺商品的客观信息(价格、库存、规格、卖点等)。\n"
    "- buyer_reviews:按商品或客户关注点检索真实买家评价。当套路要求『用真实反馈打消顾虑』"
    "时调用,不要笼统空谈。\n"
    "- handoff:遇到没学过、无法按套路可靠回答的问题时(如复杂售后纠纷),转人工并生成工单。\n\n"
    "工作流程(务必遵守):\n"
    "1. 本轮开始时系统【已经自动为你取回】店长教过的示范样本(见系统提示中的『本轮自动召回的应答示范』),"
    "直接使用即可,【不要重复调用 recall_playbook】。只有当客户话题明显切换到全新领域、"
    "而手上的示范完全不相关时,才允许再调一次。\n"
    "2. 【归纳 + 举一反三】把这些样本当成一个整体,归纳出店长背后的应答策略/套路"
    "(例如:面对比价、质量疑虑、售后担忧,不要生硬回复,先挖掘需求、引用卖点与买家评价,再顺势促单)。\n"
    "3. 判断客户当前这句话(哪怕换了问法)属于哪种情况(商品咨询/比价/物流售后/砍价),套用最匹配的示范;"
    "【即使没有完全对应的样本,也要遵循你归纳出的策略来应对,不要因为『这句没被专门教过』就轻易转人工】。\n"
    "4. 【结合完整对话上下文,不要把每句话当成孤立问题】先判断你已经了解到客户哪些信息"
    "(用途/预算/型号偏好/使用场景等):套路所需信息不足时先追问挖掘、不急着推商品,"
    "信息足够时再按策略推进。回答要自然承接上下文。\n"
    "5. 【事实红线】商品的客观事实(具体价格数字、库存、参数、能否退换等承诺)必须来自 product_search 或"
    "店长教过的样本,【绝不可自己编造】;需要而查不到时才 handoff。\n"
    "6. 只有在下列情况才调用 handoff:套路库为空(还没教过任何东西)、问题明显超出你能归纳应对的范围"
    "(如复杂售后纠纷)、或需要你没有的客观事实。\n"
    "7. 用自然、口语化的中文回复,不要暴露上述内部流程和工具名。"
)


def handle_chat(message: str, session_id: str = "demo", log_metrics: bool = True) -> LoopResult:
    # system 每轮由 Loop 注入最新的 effective_system_prompt(总纲/业务设定实时生效),
    # 会话存储只保留纯对话轮次。
    history = _SESSIONS.get(session_id, [])

    result = _LOOP.run(message, history=history, system=effective_system_prompt(),
                       preload_tool="recall_playbook")

    # 只把用户与最终回复(不含中间的 tool_calls / tool 消息)落进会话上下文。
    # 滑动窗口:只保留最近 SESSION_MAX_MESSAGES 条,历史永不失控变长——
    # 修复"每轮无条件全量拼历史,越聊越长、注意力稀释"的上下文管理问题。
    full_history = [
        m for m in result.session_messages
        if m.get("role") == "user"
        or (m.get("role") == "assistant" and not m.get("tool_calls"))
    ]
    _SESSIONS[session_id] = full_history[-config.SESSION_MAX_MESSAGES:]

    # 验收跑分不算真实经营流量,不写 metrics_log,避免污染转人工率/命中率
    if log_metrics:
        _log_metrics(session_id, message, result)
    return result


def _log_metrics(session_id: str, question: str, result: LoopResult) -> None:
    hit = 1 if (result.kb_hit and not result.handoff) else 0
    with cursor() as cur:
        cur.execute(
            "INSERT INTO metrics_log (session_id, question, hit_kb, handoff) VALUES (?, ?, ?, ?)",
            (session_id, question, hit, 1 if result.handoff else 0),
        )


def reset_session(session_id: str = "demo") -> None:
    _SESSIONS.pop(session_id, None)


def refine_reply(question: str, reply: str, feedback: str, session_id: str = "demo") -> str:
    """实测纠偏:店长对 AI 刚才这条回复提意见,AI 据此重答这一句(不写库,仅当场修正)。

    重答会遵守当前系统提示词(含总纲/铁律),并把会话里最后一条 assistant 替换为修正版,
    使后续对话从修正后的语境继续。
    """
    messages = [
        {"role": "system", "content": effective_system_prompt()},
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": (
            "以上是你刚才对客户说的话。下面是【教练点评】(是我这个培训你的人给的纠偏意见,"
            "不是客户说的,不要当成客户的话来回应):\n" + feedback
            + "\n\n请你据此,把刚才发给客户的这句话重新说一遍。只输出修正后、要直接发给客户的内容,不要解释。")},
    ]
    decision = _LOOP.llm.chat(messages, [])
    revised = (decision.content or "").strip() or reply

    hist = _SESSIONS.get(session_id)
    if hist and hist[-1].get("role") == "assistant":
        hist[-1]["content"] = revised
    return revised


def commit_refinement(question: str, answer: str, feedback: str) -> dict:
    """把纠偏结果固化为一条套路样本,并自动合并进总纲(状态合并)。"""
    from . import playbook_service
    note = "(实测纠偏)" + feedback if feedback else "(实测纠偏)"
    sample_id = playbook_service.add_sample(question, answer, note, source="refine")
    summary = regenerate_summary()
    return {"ok": True, "sample_id": sample_id, "summary": summary}


_EMPTY_SUMMARY = "套路库还是空的。去『教学模式』教它几条,我就能帮你归纳出套路总纲了。"


def induce_playbook(existing: str = "") -> str:
    """从训练样本归纳『应答套路总纲』。

    existing 非空时执行【状态合并】:完整保留店长已改写的总纲规则,
    只把样本里体现出、但现有总纲还没覆盖的新套路补进去;冲突以店长现有总纲为准。
    """
    from . import playbook_service
    samples = playbook_service.recall_all()
    if not samples:
        return _EMPTY_SUMMARY
    body = "\n\n".join(
        f"{i + 1}. 客户问题:{s['question']}\n   标准答案:{s['answer']}\n   这么答的原因/套路:{s.get('note', '') or '(未填)'}"
        for i, s in enumerate(samples)
    )
    if existing.strip():
        messages = [
            {"role": "system", "content": (
                "你是资深销售培训师。店长已经有一份【现有套路总纲】,其中可能有店长手动修订或补充的规则,"
                "这些必须尊重并原样保留。现在给你一批教学样本。请在【完整保留店长现有总纲里的规则和措辞倾向】"
                "的前提下,把样本里体现出、但现有总纲还没覆盖的新套路补充进去;若样本与店长现有总纲有冲突,"
                "一律以店长现有总纲为准。输出更新后的完整总纲,条目化、简洁;不要删除店长已有的规则,不要复述原始样本。")},
            {"role": "user", "content": (
                "【现有套路总纲(店长已改写,须保留)】:\n" + existing.strip()
                + "\n\n【全部教学样本】:\n" + body)},
        ]
    else:
        messages = [
            {"role": "system", "content": (
                "你是资深销售培训师。下面是店长教给『AI 商品咨询顾问』的一批示范"
                "(客户问题 + 标准答案 + 这么答的原因)。请你从这些具体示范中,归纳提炼出店长的"
                "『应答套路总纲』:用简洁的条目列出通用的销售策略原则(面对哪类问题该怎么应对、"
                "先做什么后做什么、什么时候先挖需求、什么时候引导等)。只输出总纲本身,不要复述原始样本。")},
            {"role": "user", "content": "以下是全部示范样本:\n\n" + body},
        ]
    decision = _LOOP.llm.chat(messages, [])
    return (decision.content or "").strip() or "(暂未归纳出内容)"


def get_or_build_summary() -> str:
    """返回已保存的总纲;若从未生成过,则首次自动归纳并存下来(保留后续编辑)。"""
    from . import settings_service
    saved = settings_service.get_summary()
    if saved.strip():
        return saved
    fresh = induce_playbook()
    if fresh and fresh != _EMPTY_SUMMARY:
        settings_service.set_summary(fresh)
    return fresh


def regenerate_summary() -> str:
    """状态合并:以店长改写后的现有总纲为基底,融合全部样本,重新归纳并保存。"""
    from . import settings_service
    merged = induce_playbook(existing=settings_service.get_summary())
    if merged and merged != _EMPTY_SUMMARY:
        settings_service.set_summary(merged)
    return merged
