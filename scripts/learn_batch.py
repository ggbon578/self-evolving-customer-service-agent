#!/usr/bin/env python3
"""模拟「自学习」产出一批候选样本(进待验证区,不直接生效)。

真实场景里这批样本来自:差评聚类、工单复盘、模型对未答问题的自我归纳。
本脚本把它们从 JSON 读进来写库,status=pending,交给验收流程把关。

用法:
    python scripts/learn_batch.py --batch b20261127 --file data/learn_candidates.json
    python scripts/learn_batch.py --batch b20261127 --file ... --activate   # 跳过闸门(仅调试)

候选 JSON 格式:
    [{"question": "...", "answer": "...", "note": "...", "source": "auto"}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import init_db                        # noqa: E402
from app.services import playbook_service         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="写入一批待验证的学习样本")
    ap.add_argument("--batch", required=True, help="批次号,例如 b20261127")
    ap.add_argument("--file", required=True, help="候选样本 JSON")
    ap.add_argument("--activate", action="store_true", help="直接生效(仅调试用,正常应走验收)")
    args = ap.parse_args()

    init_db()
    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    status = "active" if args.activate else "pending"
    ids = []
    for it in items:
        sid = playbook_service.add_sample(
            it["question"], it["answer"], it.get("note", ""),
            source=it.get("source", "auto"), status=status, batch_id=args.batch,
        )
        ids.append(sid)

    print(f"批次 {args.batch}:写入 {len(ids)} 条,状态={status},ids={ids}")
    if status == "pending":
        print("下一步:python scripts/eval.py --batch %s" % args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
