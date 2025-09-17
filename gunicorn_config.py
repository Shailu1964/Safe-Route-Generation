# Gunicorn configuration file optimized for low-memory environments
workers = 1
worker_class = 'gthread'
threads = 2
bind = '0.0.0.0:10000'
timeout = 300
keepalive = 5
max_requests = 1000
max_requests_jitter = 50