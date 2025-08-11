# Gunicorn configuration file
bind = "0.0.0.0:10000"
workers = 1
timeout = 120  # Increase timeout to 2 minutes for ML predictions
keepalive = 2
max_requests = 1000
max_requests_jitter = 50
worker_class = "sync"
worker_connections = 1000
