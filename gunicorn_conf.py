import multiprocessing
import os

# 工作进程数，通常设置为CPU核心数的2-4倍
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 1))
# 每个工作进程的线程数
threads = int(os.getenv("THREADS", 1))
# 绑定的IP和端口
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8086')}"
# 工作模式
worker_class = "uvicorn.workers.UvicornWorker"
# 超时时间
timeout = 120
# 保持连接的秒数
keepalive = 5
# 日志级别
loglevel = os.getenv("LOG_LEVEL", "info")
# 是否后台运行
daemon = False
# 访问日志格式
accesslog = "-"
# 错误日志
errorlog = "-"