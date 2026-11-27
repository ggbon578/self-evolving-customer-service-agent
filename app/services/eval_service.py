"""自学习验收服务:评估集跑分、退步判定、台账留痕。

为什么需要它:自学习是「模型自己改自己的知识」,没有验收就等于没有刹车——
可能学进一条错话术,线上从此一直说错。所以每次学习产出都必须过三道关:

  1. 固定评估集(20 题,覆盖咨询/比价/物流/售后/转人工/超范围),
     同一套题反复跑,分数才可比;
  2. 退步闸门:综合分比上一轮跌超 EVAL_REGRESS_TOLERANCE 就判 regress,
     这批样本不予生效;
  3. 台账:每次跑分写一行,谁在什么时候学了什么、分数多少、有没有回滚,全可追。

判分不调模型(避免"用模型评模型"的不稳定),而是规则化:
转人工是否符合预期、应答要点是否命中、事实红线是否触碰、回复是否过长。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .. import config
from ..db import cursor

EVAL_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_set.json"

# 单项权重:转人工判断最重(转错=客户流失或白白浪费人力),
# 其次是应答要点,再次事实红线,最后简洁度。
_W_HANDOFF = 0.35
_W_CONTENT = 0.30
_W_SAFETY = 0.20
_W_CONCISE = 0.15


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else EVAL_SET_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"] if isinstance(data, dict) else data


def judge(case: dict[str, Any], reply: str, handoff: bool) -> dict[str, Any]:
    """对单题判分,返回 4 项检查 + 本题得分。"""
    reply = reply or ""
    expect = case.get("expect_handoff", "any")

    if expect is True:
        ok_handoff = bool(handoff)
    elif expect is False:
        ok_handoff = not handoff
    else:
        ok_handoff = True

    must = case.get("must_any") or []
    # 已转人工的题不考核话术要点:工单已建,客户会由人跟进
    ok_content = True if (not must or handoff) else any(k in reply for k in must)

    forbid = case.get("forbid") or []
    hits = [f for f in forbid if f in reply]
    ok_safety = not hits

    max_chars = int(case.get("max_chars", 200))
    ok_concise = len(reply) <= max_chars

    score = (
        _W_HANDOFF * ok_handoff
        + _W_CONTENT * ok_content
        + _W_SAFETY * ok_safety
        + _W_CONCISE * ok_concise
    )
    return {
        "checks": {
            "handoff": ok_handoff,
            "content": ok_content,
            "safety": ok_safety,
            "concise": ok_concise,
        },
        "violations": hits,
        "chars": len(reply),
        "score": round(score, 3),
        "passed": all([ok_handoff, ok_content, ok_safety, ok_concise]),
    }


def decide_verdict(score: float, prev_score: float | None,
                   tolerance: float) -> tuple[str, float | None]:
    """退步判定:返回 (结论, 与上一轮的差值)。

    抽成纯函数,便于单测——闸门本身必须可靠,不能靠"跑一次看看"。
    """
    if prev_score is None:
        return "baseline", None
    delta = round(score - prev_score, 3)
    return ("regress" if delta < -tolerance else "pass"), delta


def run_eval(batch_id: str | None = None, note: str = "",
             limit: int | None = None, cases_path: str | None = None) -> dict[str, Any]:
    """跑一遍评估集,返回逐题明细 + 汇总 + 与上一轮对比的结论。

    每题用独立 session,互不影响;不写 metrics_log,避免污染线上经营指标。
    """
    from . import chat_service, playbook_service

    cases = load_cases(cases_path)
    if limit:
        cases = cases[:limit]

    # 影子放行:带批次号时,让这批"还没转正"的样本临时参与召回,
    # 测的才是"它生效之后的样子";跑完无论如何都撤销,不在库里留痕。
    shadowed = 0
    shadow_summary: str | None = None
    if batch_id:
        pending = playbook_service.pending_samples(batch_id)
        if pending:
            playbook_service.shadow_allow(batch_id)
            shadowed = len(pending)
            # 更关键的一层:新样本会重新归纳「套路总纲」,而总纲是全局注入的,
            # 一旦被带歪,影响的是所有回答(不只是相似的那几题)。所以影子期间
            # 也把总纲重算一遍——只放进内存,不写 settings。
            try:
                from . import settings_service
                shadow_summary = chat_service.induce_playbook(
                    existing=settings_service.get_summary())
                chat_service.set_summary_override(shadow_summary)
            except Exception as e:
                shadow_summary = None
                print(f"[warn] 影子总纲归纳失败,沿用现总纲: {e}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        for c in cases:
            sid = f"eval-{c['id']}-{int(t0)}"
            err = None
            reply, handoff = "", False
            try:
                res = chat_service.handle_chat(c["question"], session_id=sid, log_metrics=False)
                reply, handoff = res.reply or "", bool(res.handoff)
            except Exception as e:  # 单题异常不能拖垮整轮验收
                err = f"{type(e).__name__}: {e}"
            verdict = judge(c, reply, handoff)
            rows.append({
                "id": c["id"],
                "category": c.get("category", ""),
                "question": c["question"],
                "expect_handoff": c.get("expect_handoff"),
                "got_handoff": handoff,
                "reply": reply,
                "error": err,
                **verdict,
            })
            chat_service.reset_session(sid)
    finally:
        if shadowed:
            chat_service.set_summary_override(None)
            playbook_service.shadow_clear()

    elapsed = round(time.time() - t0, 1)
    summary = _summarize(rows)
    prev = last_run()
    tolerance = config.EVAL_REGRESS_TOLERANCE
    verdict, delta = decide_verdict(summary["score"],
                                    None if prev is None else prev["score"], tolerance)

    run_id = _save_run(batch_id, len(rows), summary["passed"], summary["score"],
                       summary["metrics"], verdict, delta, note or summary["hint"])

    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "score": summary["score"],
        "passed": summary["passed"],
        "n_cases": len(rows),
        "metrics": summary["metrics"],
        "verdict": verdict,
        "delta": delta,
        "tolerance": tolerance,
        "prev_score": None if prev is None else prev["score"],
        "elapsed_sec": elapsed,
        "shadowed": shadowed,
        "shadow_summary": shadow_summary,
        "rows": rows,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    passed = sum(1 for r in rows if r["passed"])
    score = round(sum(r["score"] for r in rows) / n, 3)

    # 分指标只统计「有该类题目」的项,避免分母为 0 产生误导性百分比
    def rate(key: str, only_explicit: bool = False) -> float:
        pool = [r for r in rows
                if not only_explicit or r["expect_handoff"] in (True, False)]
        if not pool:
            return 1.0
        return round(sum(1 for r in pool if r["checks"][key]) / len(pool), 3)

    metrics = {
        "handoff_acc": rate("handoff", only_explicit=True),
        "content_hit": rate("content"),
        "fact_safety": rate("safety"),
        "concise_rate": rate("concise"),
        "pass_rate": round(passed / n, 3),
    }
    # 失败题按类别归堆,方便定位"这次学习把哪类能力带偏了"
    fails = [r["id"] for r in rows if not r["passed"]]
    hint = ("全项通过" if not fails else "未通过:" + ",".join(fails))
    return {"score": score, "passed": passed, "metrics": metrics, "hint": hint}


def _save_run(batch_id: str | None, n_cases: int, passed: int, score: float,
              metrics: dict[str, Any], verdict: str, delta: float | None,
              note: str) -> int:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO eval_runs (batch_id, n_cases, passed, score, metrics, verdict, delta, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (batch_id, n_cases, passed, score, json.dumps(metrics, ensure_ascii=False),
             verdict, delta, note),
        )
        return cur.lastrowid


def last_run(before_id: int | None = None) -> dict[str, Any] | None:
    """取上一轮台账记录(用于对比基线)。"""
    sql = "SELECT * FROM eval_runs"
    args: tuple = ()
    if before_id:
        sql += " WHERE id<?"
        args = (before_id,)
    with cursor() as cur:
        cur.execute(sql + " ORDER BY id DESC LIMIT 1", args)
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d.get("metrics") or "{}")
        return d


def history(limit: int = 20) -> list[dict[str, Any]]:
    """台账列表:最近 N 次跑分,给后台「验收记录」页展示。"""
    with cursor() as cur:
        cur.execute("SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["metrics"] = json.loads(d.get("metrics") or "{}")
            out.append(d)
        return out
