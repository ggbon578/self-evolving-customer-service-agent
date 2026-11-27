#!/usr/bin/env python3
"""自学习验收 CLI。

用法(在项目根目录执行):
    python scripts/eval.py                      # 跑一遍 20 题,记台账,打印报告
    python scripts/eval.py --limit 3            # 冒烟:只跑前 3 题
    python scripts/eval.py --batch b20261127    # 针对某学习批次跑验收
    python scripts/eval.py --batch b1 --activate  # 跑分通过则自动把该批样本转正
    python scripts/eval.py --rollback b1        # 一键回滚该批次
    python scripts/eval.py --history            # 看历史台账
    python scripts/eval.py --json out.json      # 同时把明细存成 JSON

退出码:0 通过/基线;2 退步(regress);1 执行出错。CI 里可直接据此拦截发布。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import init_db                      # noqa: E402
from app.services import eval_service, playbook_service  # noqa: E402


def _bar(rate: float, width: int = 20) -> str:
    filled = int(round(rate * width))
    return "█" * filled + "·" * (width - filled)


def print_report(result: dict) -> None:
    m = result["metrics"]
    print("\n" + "=" * 62)
    print(f"验收报告  run#{result['run_id']}"
          + (f"  批次 {result['batch_id']}" if result["batch_id"] else ""))
    print("=" * 62)
    print(f"{'ID':<5}{'类别':<10}{'转人工':<8}{'本题':<6}未通过项")
    print("-" * 62)
    for r in result["rows"]:
        bad = [k for k, v in r["checks"].items() if not v]
        if r["error"]:
            bad.append("异常")
        got = "是" if r["got_handoff"] else "否"
        mark = "✓" if r["passed"] else "✗"
        print(f"{r['id']:<5}{r['category']:<10}{got:<8}{r['score']:<6}"
              f"{(','.join(bad) or '-')}")
    print("-" * 62)
    print(f"综合分   {result['score']:.3f}   {_bar(result['score'])}")
    print(f"全项通过 {result['passed']}/{result['n_cases']}")
    print(f"转人工判断 {m['handoff_acc']:.1%}   应答要点 {m['content_hit']:.1%}   "
          f"事实红线 {m['fact_safety']:.1%}   简洁度 {m['concise_rate']:.1%}")
    print(f"耗时 {result['elapsed_sec']}s")
    if result.get("shadowed"):
        print(f"影子放行 {result['shadowed']} 条待验证样本参与本轮召回(未落库生效)")
        if result.get("shadow_summary"):
            one = " / ".join(
                ln.strip() for ln in result["shadow_summary"].splitlines() if ln.strip())
            print("影子总纲(仅本次跑分):" + one[:110] + ("…" if len(one) > 110 else ""))

    prev = result["prev_score"]
    if prev is None:
        print("结论:首跑基线,已记入台账")
    else:
        d = result["delta"]
        arrow = "↑" if d >= 0 else "↓"
        print(f"对比上一轮 {prev:.3f} → {result['score']:.3f} ({arrow}{abs(d):.3f}, "
              f"容忍阈值 {result['tolerance']})")
        print({"pass": "结论:通过,可生效",
               "regress": "结论:退步超阈值,【拒绝生效】建议排查或回滚",
               "baseline": "结论:基线"}[result["verdict"]])


def main() -> int:
    ap = argparse.ArgumentParser(description="自学习验收")
    ap.add_argument("--batch", help="学习批次号")
    ap.add_argument("--limit", type=int, help="只跑前 N 题(冒烟)")
    ap.add_argument("--activate", action="store_true", help="跑分通过后自动转正该批次")
    ap.add_argument("--rollback", metavar="BATCH", help="一键回滚指定批次")
    ap.add_argument("--history", action="store_true", help="查看历史台账")
    ap.add_argument("--json", metavar="PATH", help="明细导出路径")
    ap.add_argument("--note", default="", help="本次验收备注")
    args = ap.parse_args()

    init_db()

    if args.rollback:
        res = playbook_service.rollback_batch(args.rollback, reason="cli")
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res["ok"] else 1

    if args.history:
        rows = eval_service.history()
        if not rows:
            print("台账为空,先跑一次 python scripts/eval.py")
            return 0
        print(f"{'run':<5}{'时间':<20}{'分数':<8}{'通过':<8}{'结论':<10}批次")
        for r in rows:
            print(f"{r['id']:<5}{r['created_at']:<20}{r['score']:<8.3f}"
                  f"{r['passed']}/{r['n_cases']:<6}{r['verdict']:<10}{r['batch_id'] or '-'}")
        return 0

    result = eval_service.run_eval(batch_id=args.batch, note=args.note, limit=args.limit)
    print_report(result)

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"明细已写入 {args.json}")

    # 退步时绝不自动生效:人工确认后再决定改样本还是回滚
    if result["verdict"] == "regress":
        if args.batch:
            print(f"提示:批次 {args.batch} 保持待验证状态;确认要撤可执行 "
                  f"python scripts/eval.py --rollback {args.batch}")
        return 2

    if args.activate and args.batch:
        res = playbook_service.activate_batch(args.batch, note=f"eval#{result['run_id']} 通过")
        print(f"已转正批次 {args.batch}:{res['changed']} 条样本生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
