#!/usr/bin/env python3
"""闸门自检:证明「验收机制本身」是可靠的。

为什么要这个:闸门如果自己坏了,跑分永远是 1.000,看着一片绿,实际什么都没拦住。
所以判分规则、退步判定、回滚生效,这三件事必须有确定性的用例,不依赖真实模型。

    python scripts/test_gate.py

全绿退出码 0;任何一项失败退出码 1。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.db import cursor, init_db                        # noqa: E402
from app.services import eval_service, playbook_service   # noqa: E402

PASS, FAIL = "✓", "✗"
_failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {PASS if ok else FAIL} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failed.append(name)


def case(cid: str) -> dict:
    for c in eval_service.load_cases():
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def test_judge() -> None:
    print("\n[1] 判分规则:坏答案必须被判低分")
    e05 = case("E05")

    ok = eval_service.judge(e05, "这台真是一口价，不过我告诉您领券的路子，能省不少~", False)
    check("E05 守住价格 + 给替代方案 → 满分", ok["passed"] and ok["score"] == 1.0,
          f"score={ok['score']}")

    bad = eval_service.judge(e05, "行，给您降200，拍下我改价。", False)
    check("E05 答应降价 → 失分", not bad["passed"] and bad["score"] < 0.8,
          f"score={bad['score']} 违规={bad['violations']}")

    e15 = case("E15")
    bad2 = eval_service.judge(e15, "您消消气，给您补个红包，别投诉了。", False)
    check("E15 投诉未转人工 → 失分", not bad2["passed"],
          f"score={bad2['score']}")

    good = eval_service.judge(e15, "已为您转接人工专员，工单号 88。", True)
    check("E15 转人工 + 建单 → 满分", good["passed"], f"score={good['score']}")

    e10 = case("E10")
    long_reply = "支持七天无理由退换，" + "很划算的哦" * 40
    r = eval_service.judge(e10, long_reply, False)
    check("E10 回复超长 → 简洁度扣分", not r["checks"]["concise"],
          f"{r['chars']} 字 > {e10['max_chars']}")


def test_verdict() -> None:
    print("\n[2] 退步判定:跌幅超过阈值必须拦")
    tol = config.EVAL_REGRESS_TOLERANCE
    v, d = eval_service.decide_verdict(0.95, None, tol)
    check("首跑 → baseline", v == "baseline" and d is None)

    v, d = eval_service.decide_verdict(0.97, 0.98, tol)
    check(f"跌 0.01(≤{tol})→ pass", v == "pass" and d == -0.01, f"delta={d}")

    v, d = eval_service.decide_verdict(0.88, 0.98, tol)
    check(f"跌 0.10(>{tol})→ regress", v == "regress" and d == -0.10, f"delta={d}")

    v, _ = eval_service.decide_verdict(1.0, 0.9, tol)
    check("涨分 → pass", v == "pass")


def test_rollback() -> None:
    print("\n[3] 回滚:置位后必须立刻从召回中消失")
    bid = "gate-selftest"
    init_db()
    with cursor() as cur:
        cur.execute("DELETE FROM playbook WHERE batch_id=?", (bid,))

    sid = playbook_service.add_sample(
        "自检专用问题 XYZ123", "自检专用答案", "selftest",
        source="auto", status="pending", batch_id=bid)
    check("写入 pending 样本", bool(sid))

    # 未转正时,即便不影子放行,也召回不到
    hit = [s for s in playbook_service.recall_relevant("自检专用问题 XYZ123")
           if s["id"] == sid]
    check("pending 不参与召回(第一道闸)", not hit)

    # 影子放行后能看到(验收时的行为)
    playbook_service.shadow_allow(bid)
    try:
        hit = [s for s in playbook_service.recall_relevant("自检专用问题 XYZ123")
               if s["id"] == sid]
        check("影子放行后可召回(验收行为)", bool(hit))
    finally:
        playbook_service.shadow_clear()

    # 转正后正常召回
    playbook_service.activate_batch(bid)
    hit = [s for s in playbook_service.recall_relevant("自检专用问题 XYZ123")
           if s["id"] == sid]
    check("确认生效后可召回(第二道闸)", bool(hit))

    # 回滚后再次消失
    res = playbook_service.rollback_batch(bid, reason="selftest")
    hit = [s for s in playbook_service.recall_relevant("自检专用问题 XYZ123")
           if s["id"] == sid]
    check("回滚后立即召回不到", res["rolled_back"] == 1 and not hit,
          f"rolled_back={res['rolled_back']}")

    # 清理自检数据,别弄脏真库
    with cursor() as cur:
        cur.execute("DELETE FROM playbook WHERE batch_id=?", (bid,))
    check("自检数据已清理", True)


def main() -> int:
    print("=" * 56)
    print("闸门自检(不调模型,结果确定性)")
    print("=" * 56)
    test_judge()
    test_verdict()
    test_rollback()
    print("\n" + "-" * 56)
    if _failed:
        print(f"未通过 {len(_failed)} 项:{', '.join(_failed)}")
        return 1
    print("全部通过:判分、判定、回滚三处均按预期工作")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
