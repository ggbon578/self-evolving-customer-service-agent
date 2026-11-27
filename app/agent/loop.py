"""Agent Loop 引擎。

核心循环:推理 → 选工具 → 调用 → 观察结果 → 再决策 → 终止。
内置边界控制(最大步数 / 超时),并全程记录轨迹用于前端可视化。

引擎只依赖 LLMProvider 抽象与 Tool 接口,与具体模型/工具实现解耦。
"""
from __future__ import annotations
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider
from .. import config
from .tools import build_registry


# 客户主动要求转人工的表征词。命中即强制建单,不能指望模型自己判断
# (模型的默认倾向是挽留客户、"我这边可以帮您处理")。
_HANDOFF_HINTS = (
    "转人工", "人工客服", "人工服务", "找人工", "要人工", "叫人工", "换人工",
    "转接人工", "真人客服", "客服电话", "你们领导", "你们经理", "你们主管",
    "换个客服", "叫你们老板", "投诉", "12315", "消协", "工商局",
)
# 出现这些词说明客户在"问人工客服的规则",不是在"要求转人工",不触发。
_HANDOFF_NEGATIVE = ("几点", "多久", "什么时候", "几点钟", "上班", "工作时间", "有没有人工")


def _wants_human(text: str) -> bool:
    t = (text or "").replace(" ", "").replace("，", "").replace(",", "")
    if not t:
        return False
    if any(n in t for n in _HANDOFF_NEGATIVE):
        return False
    return any(h in t for h in _HANDOFF_HINTS)


@dataclass
class LoopResult:
    reply: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    handoff: bool = False
    ticket_id: int | None = None
    kb_hit: bool = False
    session_messages: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.tools = build_registry()

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def run(self, user_message: str,
            history: list[dict[str, Any]] | None = None,
            system: str | None = None,
            preload_tool: str | None = None) -> LoopResult:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages += list(history or [])
        messages.append({"role": "user", "content": user_message})

        trace: list[dict[str, Any]] = []
        handoff = False
        ticket_id: int | None = None
        kb_hit = False
        # 见下方"机器人能力边界的两个升级信号"说明
        last_call_key: str | None = None
        empty_recall_rounds = 0

        # 强制前置召回:不依赖模型自觉。每轮由引擎先查一次套路库,把命中的示范
        # 直接写进系统提示(而不是伪装成工具对话),避免模型进入"讲解套路"的元模式。
        if preload_tool and preload_tool in self.tools:
            tool = self.tools[preload_tool]
            args = {"query": user_message}
            trace.append({"step": 0, "type": "tool_call", "tool": preload_tool,
                          "input": args, "auto": True})
            try:
                result = tool.run(**args)
            except Exception as exc:
                result = {"error": f"工具 {preload_tool} 执行出错:{exc}"}
            trace.append({"step": 0, "type": "tool_result", "tool": preload_tool,
                          "output": result,
                          "is_error": isinstance(result, dict) and "error" in result})
            samples = (result or {}).get("samples") or []
            # 注入裁剪:只取最相关的前 N 条(召回结果已按相似度降序,且低分已被门槛过滤)。
            # 之前 5 条全量注入,一条样本两三百字、5 条上千字,每轮都塞进提示词,
            # 造成上下文过载、注意力稀释——正是老师指出的"文本过载"问题。
            top_samples = samples[: config.PLAYBOOK_INJECT_TOP_N]
            if top_samples:
                kb_hit = True
                blocks = []
                for i, s in enumerate(top_samples, 1):
                    answer = (s.get("answer") or "").strip()
                    if len(answer) > config.PLAYBOOK_ANSWER_MAX_CHARS:
                        answer = answer[: config.PLAYBOOK_ANSWER_MAX_CHARS] + "……"
                    score = s.get("score")
                    score_tag = f"(相似度 {score})" if score is not None else ""
                    blocks.append(
                        f"{i}. 相似的客户问题{score_tag}:{s['question']}\n"
                        f"   应答套路:{s.get('note', '') or '(未填)'}\n"
                        f"   话术参考:{answer}"
                    )
                injected = (
                    "\n\n【本轮自动召回的应答示范(内部资料,仅供你模仿,禁止向客户提及)】\n"
                    + "\n".join(blocks)
                    + "\n\n以上资料已取回完毕,无需再次检索、也不要重复调用查库工具。"
                    "使用时须遵守:以上只是你要模仿的说话方式和处理顺序,"
                    "【绝对不能】在回复里出现\"套路\"\"示范\"\"样本\"\"店长教过\"\"根据规定\""
                    "\"归纳\"\"第一步/第二步\"等内部措辞,也不要把策略解释给客户听。"
                    "像真人客服一样,直接对客户说该说的话。"
                )
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = (messages[0].get("content") or "") + injected
                else:
                    messages.insert(0, {"role": "system", "content": injected})
            else:
                # 召回为空:如实告知模型"没有匹配样本",让它按通用原则作答,
                # 涉及商品事实时只能依据工具查询结果,禁止编造。
                empty_recall_rounds = 1  # 记为第一轮空召回;循环内模型再召回仍为空则触发转人工
                injected = (
                    f"\n\n【本轮自动召回结果】套路库中没有与客户问题足够相似的样本"
                    f"(相似度门槛 {config.PLAYBOOK_MIN_SCORE})。"
                    "请按通用客服原则直接、自然地回答;涉及具体商品、价格、库存、售后政策时,"
                    "只能依据工具查询到的真实结果,查不到就如实说明,【禁止编造】。"
                )
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = (messages[0].get("content") or "") + injected
                else:
                    messages.insert(0, {"role": "system", "content": injected})

        # 客户明确要求转人工:强制建单,并明确告知模型不许再挽留。
        # 这是硬规则——模型遇到"给我转人工"时通常回"我这边可以帮您处理",属于推诿。
        if _wants_human(user_message) and "handoff" in self.tools:
            hargs = {"question": user_message, "context": "客户主动要求转接人工客服"}
            trace.append({"step": 0, "type": "tool_call", "tool": "handoff",
                          "input": hargs, "auto": True})
            try:
                hres = self.tools["handoff"].run(**hargs)
            except Exception as exc:
                hres = {"error": f"工具 handoff 执行出错:{exc}"}
            trace.append({"step": 0, "type": "tool_result", "tool": "handoff",
                          "output": hres,
                          "is_error": isinstance(hres, dict) and "error" in hres})
            if isinstance(hres, dict) and hres.get("ticket_id"):
                handoff = True
                ticket_id = hres["ticket_id"]
            tip = (
                "\n\n【最高优先级·客户已明确要求转人工】客户这句话是在要求转接人工客服,不是咨询。"
                + (f"本单已生成工单(工单号 {ticket_id})。" if ticket_id else "本单正在建单。")
                + "你必须直接回答:已为您转接人工专员、工单号多少、请稍等。"
                "【禁止】再说\"我这边可以帮您处理\"\"我帮您看看\"之类挽留或推诿的话,"
                "禁止继续追问需求、禁止推销商品、禁止解释流程。一句话交接完即可。"
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (messages[0].get("content") or "") + tip
            else:
                messages.insert(0, {"role": "system", "content": tip})

        started = time.time()
        step = 0
        reply = "抱歉,系统繁忙,请稍后再试。"

        # 机器人能力边界的两个升级信号(转人工的三条规则之二/三,与"客户明确要求"
        # 并行,哪条先来走哪条):
        # ①决策重复:同一工具+完全相同入参连调两次 → 没有新信息可拿,判定答不出;
        # ②连续两轮召回为空:套路库匹配不上 → 超出机器人能力范围。
        # 任一触发即强制建单转人工。

        def _force_handoff(reason: str) -> None:
            """引擎层强制转人工:建单 + 注入最高优先级指令,不依赖模型自觉。"""
            nonlocal handoff, ticket_id, empty_recall_rounds
            htool = self.tools.get("handoff")
            if htool is None:
                return
            try:
                hres = htool.run(question=user_message, context=reason)
            except Exception:
                hres = {}
            handoff = True
            ticket_id = hres.get("ticket_id")
            trace.append({"step": step, "type": "tool_call", "tool": "handoff",
                          "input": {"question": user_message, "context": reason},
                          "auto": True})
            trace.append({"step": step, "type": "tool_result", "tool": "handoff",
                          "output": hres, "is_error": False})
            tip = (
                f"\n\n【最高优先级·转人工】{reason}"
                + (f"本单已生成工单(工单号 {ticket_id})。" if ticket_id else "")
                + "你必须直接回答:已为您转接人工专员、工单号多少、请稍等。"
                "【禁止】挽留、推诿、继续追问需求或再尝试用工具解决,一句话交接完即可。"
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (messages[0].get("content") or "") + tip
            else:
                messages.insert(0, {"role": "system", "content": tip})
            empty_recall_rounds = 0

        while step < config.MAX_LOOP_STEPS:
            if time.time() - started > config.LOOP_TIMEOUT_SECONDS:
                reply = "处理超时,已为您转接人工客服。"
                trace.append({"step": step, "type": "final",
                              "content": "(超时,边界控制触发)"})
                break

            step += 1
            schemas = self._tool_schemas()

            # ① LLM 调用前:记录本步喂给模型的完整上下文(调试"上下文是什么")
            trace.append({"step": step, "type": "llm_call",
                          "messages": copy.deepcopy(messages),
                          "available_tools": [t["name"] for t in schemas]})

            decision = self.llm.chat(messages, schemas)

            # ② LLM 返回后:记录模型原始决策(调试"模型决定做什么")
            trace.append({"step": step, "type": "llm_response",
                          "decision": decision.type,
                          "thought": decision.thought,
                          "tool_calls": [{"id": c.id, "name": c.name, "input": c.input}
                                         for c in decision.tool_calls],
                          "content": decision.content})

            # 记录"思考"
            if decision.thought:
                trace.append({"step": step, "type": "think",
                              "content": decision.thought})

            if decision.type == "final":
                reply = decision.content or ""
                trace.append({"step": step, "type": "final", "content": reply})
                messages.append({"role": "assistant", "content": reply})
                break

            # tool_calls:模型本步可请求一个或多个工具(并行);按标准
            # assistant(发起 tool_calls)→ tool(带 tool_call_id 回填)结构记录
            calls = decision.tool_calls
            is_dup = False
            messages.append({
                "role": "assistant",
                "content": decision.thought or "",
                "tool_calls": [{"id": c.id, "name": c.name, "input": c.input}
                               for c in calls],
            })
            for call in calls:
                trace.append({"step": step, "type": "tool_call",
                              "tool": call.name, "input": call.input})
                tool = self.tools.get(call.name)
                call_key = json.dumps({"tool": call.name, "input": call.input or {}},
                                      ensure_ascii=False, sort_keys=True)

                # 升级信号①:决策重复——同名工具+完全相同入参连调两次,
                # 结果必然还是那些,说明没东西可拿了,判定机器人答不出。
                is_dup = call.name != "handoff" and call_key == last_call_key
                if is_dup:
                    result = {"error": ("决策重复:上一步已用完全相同的参数调用过同一工具,"
                                        "重复执行不会有任何新结果。这说明你已无法解决问题,"
                                        "不要再重试,立即调用 handoff 转人工。")}
                    trace.append({"step": step, "type": "note",
                                  "content": "决策重复检测触发:同名工具+同参数连调两次 → 判定答不出,转人工"})
                elif tool is None:
                    result = {"error": f"未知工具:{call.name}"}
                else:
                    try:
                        result = tool.run(**(call.input or {}))
                    except Exception as exc:  # 工具异常转成观测结果回填,让模型自行纠错
                        result = {"error": f"工具 {call.name} 执行出错:{exc}"}
                last_call_key = call_key

                is_error = isinstance(result, dict) and "error" in result
                trace.append({"step": step, "type": "tool_result",
                              "tool": call.name, "output": result,
                              "is_error": is_error})
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "name": call.name, "result": result})

                # 旁路记录关键信号(用于指标与返回)
                if call.name == "handoff" and isinstance(result, dict):
                    handoff = True
                    ticket_id = result.get("ticket_id")
                if call.name == "recall_playbook" and isinstance(result, dict):
                    kb_hit = kb_hit or bool(result.get("count"))
                    # 升级信号②:连续两轮召回为空 → 套路库匹配不上,超出能力范围
                    if result.get("count"):
                        empty_recall_rounds = 0
                    else:
                        empty_recall_rounds += 1
                    if empty_recall_rounds >= 2 and not handoff:
                        trace.append({"step": step, "type": "note",
                                      "content": "连续两轮召回为空 → 判定超出机器人能力范围,转人工"})
                        _force_handoff("客户的问题连续两轮在套路库中匹配不到样本,已超出机器人能力范围。")
                if is_dup and not handoff:
                    _force_handoff("机器人对同一工具以完全相同的参数重复调用,没有获得任何新信息,判定无法解决问题。")
        else:
            # while 正常结束(达到最大步数仍未 final)
            reply = "这个问题比较复杂,已为您转接人工客服跟进。"
            trace.append({"step": step, "type": "final",
                          "content": "(达到最大循环步数,边界控制触发)"})

        # 兜底:回复里已经承诺转人工,但模型没真正调用 handoff → 引擎补建工单,
        # 避免"嘴上转人工、实际没建单"导致售后问题丢失。
        if (not handoff and reply
                and any(k in reply for k in ("转人工", "转接人工", "人工专员", "人工客服"))):
            handoff_tool = self.tools.get("handoff")
            if handoff_tool is not None:
                try:
                    hres = handoff_tool.run(question=user_message,
                                            context=f"模型判断需人工介入:{reply[:120]}")
                    handoff = True
                    ticket_id = hres.get("ticket_id")
                    trace.append({"step": step, "type": "tool_call", "tool": "handoff",
                                  "input": {"question": user_message}, "auto": True})
                    trace.append({"step": step, "type": "tool_result", "tool": "handoff",
                                  "output": hres, "is_error": False})
                except Exception:
                    pass

        return LoopResult(reply=reply, trace=trace, handoff=handoff,
                          ticket_id=ticket_id, kb_hit=kb_hit,
                          session_messages=messages)
