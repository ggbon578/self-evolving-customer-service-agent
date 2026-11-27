"""启动入口:python run.py 然后浏览器打开 http://127.0.0.1:8000(服务器部署时经 Nginx 反代对外提供 80 端口服务)"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
