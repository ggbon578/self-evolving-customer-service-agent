"""FastAPI 入口:商品咨询、教学(套路库)、工单、统计,以及前端静态页。"""
from __future__ import annotations
import hashlib
import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .db import init_db
from .models import (ChatRequest, ChatResponse, TeachRequest,
                     TicketTeachRequest, SampleUpdate, DirectiveUpdate,
                     SummaryUpdate, RefineRequest, RefineCommitRequest)
from .services import (chat_service, review_service, playbook_service,
                       settings_service)

app = FastAPI(title="星辰数码官方旗舰店 · AI 商品客服 · Agent Loop + 教学自进化")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    _seed_samples()


_SEED_SAMPLES = [
    ("多久发货?",
     "亲,工作日下午 4 点前下单当天发货,之后次日发货。默认中通快递,也可备注顺丰哦,一般 2-3 天到家~",
     "物流类问题直接给时效和快递选项,体现专业,不留犹豫空间。"),
    ("可以退换货吗?",
     "放心的哈,支持七天无理由退换(商品未拆封、不影响二次销售),质量问题运费我们承担,整机一年质保。您放心买~",
     "售后类问题先给足安全感,再点出政策边界,不啰嗦。"),
    ("扫地机器人适合养宠物的家庭吗?",
     "家里有毛孩子的话很推荐 X1 Pro,5000Pa 大吸力+防缠绕滚刷,猫毛狗毛都能处理干净。方便说下家里面积多大吗?我帮您看看这款够不够用~",
     "需求不明确时先顺着给结论建立信任,再追问面积等信息,为精准推荐铺路。"),
]


def _seed_samples() -> None:
    """首次启动预置几条电商训练样本,让客服开箱即可应对常见问题。"""
    from .services import playbook_service
    if not playbook_service.list_samples():
        for q, a, n in _SEED_SAMPLES:
            playbook_service.add_sample(q, a, n)


# ---------- 简单登录(访问门禁) ----------

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")
_COOKIE = "portal_token"


def _token() -> str:
    return hashlib.sha256(f"{ADMIN_USER}:{ADMIN_PASS}:xingchen-digital".encode()).hexdigest()


_LOGIN_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>登录 · 星辰数码 AI 客服</title>
<style>body{display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-family:system-ui;background:#f2f5f9}
.card{background:#fff;padding:32px 40px;border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,.08);width:300px}
h1{font-size:18px;margin:0 0 20px}input{width:100%;padding:10px;margin-bottom:12px;border:1px solid #d5dde5;border-radius:8px;box-sizing:border-box}
button{width:100%;padding:10px;background:#2563eb;color:#fff;border:0;border-radius:8px;font-size:15px;cursor:pointer}
#err{color:#e02020;font-size:13px;height:18px;margin-bottom:6px}</style></head><body>
<div class="card"><h1>星辰数码 · AI 客服后台</h1>
<div id="err"></div><input id="u" placeholder="账号"><input id="p" type="password" placeholder="密码">
<button onclick="doLogin()">登 录</button></div>
<script>async function doLogin(){const e=document.getElementById('err');e.textContent='';
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({user:document.getElementById('u').value,pass:document.getElementById('p').value})});
const j=await r.json();if(j.ok){location.href='/'}else{e.textContent='账号或密码错误';}}</script></body></html>"""


@app.get("/login")
def login_page() -> HTMLResponse:
    return HTMLResponse(_LOGIN_HTML)


@app.get("/logout")
def logout() -> RedirectResponse:
    """退出登录:清除登录 Cookie 并回到登录页(方便测试账号密码)。"""
    resp = RedirectResponse("/login")
    resp.delete_cookie(_COOKIE)
    return resp


@app.post("/api/login")
async def api_login(request: Request) -> JSONResponse:
    data = await request.json()
    if data.get("user") == ADMIN_USER and data.get("pass") == ADMIN_PASS:
        resp = JSONResponse({"ok": True})
        resp.set_cookie(_COOKIE, _token(), max_age=7 * 24 * 3600, httponly=True)
        return resp
    return JSONResponse({"ok": False}, status_code=401)


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if path in ("/login", "/api/login", "/favicon.ico") or path.startswith("/static/"):
        return await call_next(request)
    if request.cookies.get(_COOKIE) == _token():
        return await call_next(request)
    return RedirectResponse("/login")


# ---------- 咨询(客户视角) ----------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = chat_service.handle_chat(req.message, req.session_id or "demo")
    return ChatResponse(
        reply=result.reply,
        handoff=result.handoff,
        ticket_id=result.ticket_id,
        trace=result.trace,
    )


@app.post("/api/session/reset")
def reset_session(session_id: str = "demo") -> dict:
    chat_service.reset_session(session_id)
    return {"ok": True}


# ---------- 实测纠偏(点评式教学) ----------

@app.post("/api/refine")
def refine(req: RefineRequest) -> dict:
    """据店长点评,把 AI 刚才这条回复当场重答一遍。"""
    revised = chat_service.refine_reply(req.question, req.reply, req.feedback, req.session_id)
    return {"reply": revised}


@app.post("/api/refine/commit")
def refine_commit(req: RefineCommitRequest) -> dict:
    """把纠偏结果固化为套路样本,并自动合并进总纲。"""
    return chat_service.commit_refinement(req.question, req.answer, req.feedback)


# ---------- 教学 / 套路库(训练样本) ----------

@app.post("/api/teach")
def teach(req: TeachRequest) -> dict:
    sample_id = playbook_service.add_sample(req.question, req.answer, req.note)
    return {"ok": True, "sample_id": sample_id}


@app.get("/api/playbook")
def get_playbook(source: str | None = None) -> list[dict]:
    """训练样本列表;可选 source 过滤:taught(对话教学)/ refine(实测纠偏)/ ticket(工单补充)。"""
    return playbook_service.list_samples(source)


@app.get("/api/playbook/summary")
def playbook_summary() -> dict:
    """套路总纲:返回已保存版本(可被店长改写);首次访问自动归纳并保存。"""
    return {"summary": chat_service.get_or_build_summary()}


@app.put("/api/playbook/summary")
def save_playbook_summary(req: SummaryUpdate) -> dict:
    """店长手动改写总纲并保存。"""
    return settings_service.set_summary(req.text)


@app.post("/api/playbook/summary/regenerate")
def regenerate_playbook_summary() -> dict:
    """状态合并:保留店长改写,融合全部样本重新归纳。"""
    return {"summary": chat_service.regenerate_summary()}


@app.put("/api/playbook/{sample_id}")
def update_playbook(sample_id: int, req: SampleUpdate) -> dict:
    return playbook_service.update_sample(sample_id, req.question, req.answer, req.note)


@app.delete("/api/playbook/{sample_id}")
def delete_playbook(sample_id: int) -> dict:
    return playbook_service.delete_sample(sample_id)


# ---------- 工单(答不上的问题) → 教它 ----------

@app.get("/api/tickets")
def get_tickets(status: str | None = "open") -> list[dict]:
    return review_service.list_tickets(status)


@app.post("/api/tickets/{ticket_id}/teach")
def teach_ticket(ticket_id: int, req: TicketTeachRequest) -> dict:
    return review_service.teach_from_ticket(ticket_id, req.answer, req.note)


# ---------- AI 业务设定(全局人设/语气/铁律) ----------

@app.get("/api/settings/directive")
def get_directive() -> dict:
    return {"directive": settings_service.get_directive()}


@app.put("/api/settings/directive")
def put_directive(req: DirectiveUpdate) -> dict:
    return settings_service.set_directive(req.text)


@app.get("/api/settings/system-prompt")
def get_system_prompt() -> dict:
    """完整生效的系统提示词(技术流程 + 业务设定),供店长查看。"""
    return {"prompt": chat_service.effective_system_prompt()}


# ---------- 统计 ----------

@app.get("/api/metrics")
def get_metrics() -> dict:
    return review_service.metrics()


# ---------- 自学习验收(双闸 / 回滚 / 台账) ----------

class EvalRequest(BaseModel):
    batch_id: str | None = None
    limit: int | None = None
    note: str = ""


@app.get("/api/learn/pending")
def learn_pending(batch_id: str | None = None) -> list[dict]:
    """待验证区:学过但还没生效的样本。"""
    return playbook_service.pending_samples(batch_id)


@app.get("/api/learn/batches")
def learn_batches() -> list[dict]:
    """学习批次一览(这批多少条、什么状态)。"""
    return playbook_service.batches()


@app.get("/api/learn/approved-count")
def learn_approved_count() -> dict:
    """已生效样本数(双闸口径);与 /api/metrics 的 kb_count 对齐校验用。"""
    return {"active": playbook_service.count_active()}


@app.post("/api/learn/sample/{sample_id}/activate")
def learn_activate(sample_id: int) -> dict:
    """第二道闸:店长逐条确认生效。"""
    return playbook_service.activate_sample(sample_id)


@app.post("/api/learn/batch/{batch_id}/activate")
def learn_activate_batch(batch_id: str) -> dict:
    return playbook_service.activate_batch(batch_id, note="manual")


@app.post("/api/learn/batch/{batch_id}/rollback")
def learn_rollback(batch_id: str) -> dict:
    """一键回滚:整批样本立即从召回中消失(数据保留可复盘)。"""
    return playbook_service.rollback_batch(batch_id, reason="manual")


@app.post("/api/eval/run")
def eval_run(req: EvalRequest) -> dict:
    """在后台点一下就跑一遍 20 题验收。注意:真实模型约 1~3 分钟。"""
    from .services import eval_service
    result = eval_service.run_eval(batch_id=req.batch_id, note=req.note, limit=req.limit)
    # 明细里的 reply 很长,接口只回汇总,明细去 /api/eval/history 或用 CLI 导出
    return {k: v for k, v in result.items() if k != "rows"}


@app.get("/api/eval/history")
def eval_history(limit: int = 20) -> list[dict]:
    from .services import eval_service
    return eval_service.history(limit)


# ---------- 前端 ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
