#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

unit=probplots-web.service
cat > "$unit" <<'UNIT'
[Unit]
Description=probplots-web (FastAPI)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/zk/projects/probplots-web
Environment=OPENBLAS_NUM_THREADS=1
ExecStart=/home/zk/projects/probplots-web/.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 5009
Restart=always
RestartSec=2
User=zk
Group=zk

[Install]
WantedBy=multi-user.target
UNIT

sudo /usr/bin/install -m 0644 "$unit" /etc/systemd/system/$unit
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl enable --now $unit
sudo /usr/bin/systemctl status --no-pager $unit || true
