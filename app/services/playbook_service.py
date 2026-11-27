"""套路库(训练样本)服务。

一条训练样本 = 店长教的一次示范:客户问题 + 标准答案 + 这么答的原因(套路)。
服务时按客户当前问句做 embedding 语义检索,只召回最相关的 top-k 条样本交给
大模型精判并模仿作答 —— 库变大也不会撑爆上下文。总纲归纳仍取全量(recall_all)。
"""
from __future__ import annotations
import json
from typing import Any

from ..db import cursor
from .. import config
from ..knowledge import embedder


def add_sample(question: str, answer: str, note: str = "",
               source: str = "taught", status: str | None = None,
               batch_id: str | None = None) -> int:
    """写入一条训练样本。

    status='pending' 用于自学习通道:先进待验证区,跑分通过 + 店长确认后才生效,
    避免模型/脚本自己「学歪」的东西直接污染线上召回。
    status 不传时按配置决定:默认即时生效(保持人工教学的原有体验)。
    """
    if status is None:
        status = "pending" if config.PLAYBOOK_REQUIRE_APPROVAL else "active"
    vector, sig = embedder.embed(question)
    with cursor() as cur:
        cur.execute(
            "INSERT INTO playbook (question, answer, note, source, vector, vec_model, status, batch_id, activated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ?='active' THEN datetime('now','localtime') ELSE NULL END)",
            (question, answer, note, source, json.dumps(vector), sig, status, batch_id, status),
        )
        return cur.lastrowid


def list_samples(source: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """列出训练样本;可叠加 source(taught/refine/ticket)与 status(active/pending/rolled_back)过滤。"""
    sql = "SELECT * FROM playbook"
    where, args = [], []
    if source:
        where.append("source=?")
        args.append(source)
    if status:
        where.append("status=?")
        args.append(status)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC"
    with cursor() as cur:
        cur.execute(sql, tuple(args))
        return [dict(r) for r in cur.fetchall()]


# 验收「影子放行」:跑分时需要先看这批候选样本进来后会怎样,但此时它们还没转正。
# 用进程内集合临时放行,不改库状态——线上召回照旧,跑完 finally 清掉。
# 这样既能测出"学了会变好还是变坏",又不会在验收期间把未验证的东西真的推给客户。
_EXTRA_ACTIVE_BATCHES: set[str] = set()


def shadow_allow(batch_id: str) -> None:
    _EXTRA_ACTIVE_BATCHES.add(batch_id)


def shadow_clear() -> None:
    _EXTRA_ACTIVE_BATCHES.clear()


def _active_cond() -> tuple[str, tuple]:
    """召回用的生效条件;有影子批次时一并放行(仅当前进程)。"""
    batches = sorted(_EXTRA_ACTIVE_BATCHES)
    if not batches:
        return "status='active'", ()
    placeholders = ",".join("?" * len(batches))
    return f"(status='active' OR batch_id IN ({placeholders}))", tuple(batches)


def recall_all() -> list[dict[str, Any]]:
    """召回全部【已生效】样本(供总纲归纳用)。

    只取 active:待验证/已回滚的样本不该影响总纲,否则双闸形同虚设。
    验收时若开了影子放行,该批次也会被计入——测的就是"加上它之后的效果"。
    """
    cond, args = _active_cond()
    with cursor() as cur:
        cur.execute(
            f"SELECT id, question, answer, note FROM playbook WHERE {cond} ORDER BY id",
            args,
        )
        return [dict(r) for r in cur.fetchall()]


def _sample_vector(row: dict[str, Any], sig: str) -> list[float]:
    """取样本问句向量;缺失或后端签名不一致(切了 embedding 后端)则重算并回写。"""
    if row.get("vector") and row.get("vec_model") == sig:
        try:
            return json.loads(row["vector"])
        except (json.JSONDecodeError, TypeError):
            pass
    vector, new_sig = embedder.embed(row["question"])
    with cursor() as cur:
        cur.execute("UPDATE playbook SET vector=?, vec_model=? WHERE id=?",
                    (json.dumps(vector), new_sig, row["id"]))
    return vector


def recall_topk(query: str, k: int) -> list[dict[str, Any]]:
    """按客户问句语义检索,返回最相关的 top-k 条样本(按相似度降序,带 score)。

    query 为空时退化为全量返回,避免空检索导致答不上。
    """
    cond, args = _active_cond()
    with cursor() as cur:
        cur.execute(
            f"SELECT id, question, answer, note, vector, vec_model FROM playbook "
            f"WHERE {cond} ORDER BY id",
            args,
        )
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return []
    if not (query or "").strip():
        return [{"id": r["id"], "question": r["question"], "answer": r["answer"],
                 "note": r["note"]} for r in rows]

    sig = embedder.signature()
    qvec, _ = embedder.embed(query)
    scored = []
    for r in rows:
        score = embedder.cosine(qvec, _sample_vector(r, sig))
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    # 质量门槛:低于最低相似度的样本视为无关,一律不返回。
    # 之前只排序取前 k 条,导致 0 分(完全无关)样本也被硬凑进结果喂给模型;
    # 现在宁缺毋滥——凑不满 k 条就少返回几条,甚至返回空,由上层如实告知模型。
    return [{"id": r["id"], "question": r["question"], "answer": r["answer"],
             "note": r["note"], "score": round(score, 3)}
            for score, r in scored if score >= config.PLAYBOOK_MIN_SCORE][:k]


def _trim(sample: dict[str, Any]) -> dict[str, Any]:
    """裁剪单条样本:话术、策略都截断到上限——瘦上下文,保精髓。"""
    s = dict(sample)
    if len(s.get("answer") or "") > config.PLAYBOOK_ANSWER_MAX_CHARS:
        s["answer"] = s["answer"][: config.PLAYBOOK_ANSWER_MAX_CHARS] + "……"
    if len(s.get("note") or "") > config.PLAYBOOK_NOTE_MAX_CHARS:
        s["note"] = s["note"][: config.PLAYBOOK_NOTE_MAX_CHARS] + "……"
    return s


def recall_relevant(query: str) -> list[dict[str, Any]]:
    """对外的主召回入口:门槛过滤 + 只取最相关 top-N 条 + 话术裁剪。

    与注入提示词的条数对齐,工具输出和模型看到的示范保持一致,
    轨迹里也不会再挂着一长串完整样本。
    """
    return [_trim(s) for s in recall_topk(query, config.PLAYBOOK_INJECT_TOP_N)]


def update_sample(sample_id: int, question: str, answer: str, note: str) -> dict[str, Any]:
    vector, sig = embedder.embed(question)
    with cursor() as cur:
        cur.execute(
            "UPDATE playbook SET question=?, answer=?, note=?, vector=?, vec_model=? WHERE id=?",
            (question, answer, note, json.dumps(vector), sig, sample_id),
        )
    return {"ok": True}


def delete_sample(sample_id: int) -> dict[str, Any]:
    with cursor() as cur:
        cur.execute("DELETE FROM playbook WHERE id=?", (sample_id,))
    return {"ok": True}


def count() -> int:
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM playbook")
        return cur.fetchone()["c"]


def count_active() -> int:
    """已生效(会被召回)的样本数——双闸后的真实知识量。"""
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM playbook WHERE status='active'")
        return cur.fetchone()["c"]


# ---------- 双闸:待验证 → 确认生效 / 整体回滚 ----------

def activate_sample(sample_id: int) -> dict[str, Any]:
    """第二道闸:店长确认后,把单条待验证样本转正。"""
    with cursor() as cur:
        cur.execute(
            "UPDATE playbook SET status='active', activated_at=datetime('now','localtime') "
            "WHERE id=? AND status='pending'",
            (sample_id,),
        )
        changed = cur.rowcount
    return {"ok": bool(changed), "sample_id": sample_id, "changed": changed}


def activate_batch(batch_id: str, note: str = "") -> dict[str, Any]:
    """整批转正:一次学习产出的所有样本同时生效。"""
    with cursor() as cur:
        cur.execute(
            "UPDATE playbook SET status='active', activated_at=datetime('now','localtime') "
            "WHERE batch_id=? AND status='pending'",
            (batch_id,),
        )
        changed = cur.rowcount
    return {"ok": bool(changed), "batch_id": batch_id, "changed": changed, "note": note}


def rollback_batch(batch_id: str, reason: str = "manual") -> dict[str, Any]:
    """一键回滚:把某一批次学到的样本全部作废(置 rolled_back,不物理删除)。

    保留数据便于事后复盘「这批到底学了什么、为什么回滚」;
    召回只认 status='active',所以置位后立刻从线上消失。
    """
    with cursor() as cur:
        cur.execute(
            "UPDATE playbook SET status='rolled_back' WHERE batch_id=? AND status IN ('active','pending')",
            (batch_id,),
        )
        changed = cur.rowcount
    return {"ok": bool(changed), "batch_id": batch_id, "rolled_back": changed, "reason": reason}


def pending_samples(batch_id: str | None = None) -> list[dict[str, Any]]:
    """待验证区样本(给后台「待确认」列表用)。"""
    sql = "SELECT id, question, answer, note, source, batch_id, created_at FROM playbook WHERE status='pending'"
    args: tuple = ()
    if batch_id:
        sql += " AND batch_id=?"
        args = (batch_id,)
    with cursor() as cur:
        cur.execute(sql + " ORDER BY id DESC", args)
        return [dict(r) for r in cur.fetchall()]


def batches() -> list[dict[str, Any]]:
    """批次一览:每批多少条、什么状态,便于逐批验收/回滚。"""
    with cursor() as cur:
        cur.execute(
            "SELECT batch_id, status, COUNT(*) AS n, MIN(created_at) AS created_at "
            "FROM playbook WHERE batch_id IS NOT NULL GROUP BY batch_id, status ORDER BY batch_id DESC"
        )
        return [dict(r) for r in cur.fetchall()]
