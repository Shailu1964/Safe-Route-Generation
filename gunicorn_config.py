# Gunicorn configuration file
workers = 4
worker_class = 'sync'
bind = '0.0.0.0:10000'
timeout = 300  # Increased from 120 to 300 seconds
worker_connections = 1000
keepalive = 5
threads = 4
max_requests = 1000
max_requests_jitter = 50