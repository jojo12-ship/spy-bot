"""
S&P 500 Telegram Signal Bot — entry point.

Reads TELEGRAM_SPY_BOT_TOKEN from environment and starts polling.
Runs a tiny health-check HTTP server on PORT so the workflow runner
is satisfied that a port was opened.

Auto-restarts on crash with exponential backoff.
"""
import logging
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

TOKEN = re.sub(r'\s+', '', os.getenv("TELEGRAM_SPY_BOT_TOKEN") or "")
if not TOKEN:
    logger.error(
        "TELEGRAM_SPY_BOT_TOKEN is not set. "
        "Set it as a Replit secret and restart."
    )
    sys.exit(1)

PORT = int(os.getenv("PORT", "8001"))


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"SPY Signal Bot is running")

    def log_message(self, *args):
        pass   # silence access logs


def _start_health_server():
    server = HTTPServer(("0.0.0.0", PORT), _HealthHandler)
    logger.info(f"Health server listening on port {PORT}")
    server.serve_forever()


# Start health server in background thread before bot polling
t = threading.Thread(target=_start_health_server, daemon=True)
t.start()

def _keep_alive():
    import urllib.request
    domain = os.getenv("REPLIT_DEV_DOMAIN", "")
    if not domain:
        return
    urls = [
        f"https://{domain}/crypto/health",
        f"https://{domain}/kalshi/health",
    ]
    while True:
        time.sleep(120)  # every 2 minutes
        for url in urls:
            try:
                urllib.request.urlopen(url, timeout=10)
            except Exception:
                pass

threading.Thread(target=_keep_alive, daemon=True).start()
logger.info("Keep-alive pinger started")

from bot import build_app

MAX_BACKOFF = 60   # seconds

attempt = 0
while True:
    attempt += 1
    try:
        app = build_app(TOKEN)
        logger.info(f"S&P 500 Signal Bot starting... (attempt {attempt})")
        app.run_polling(drop_pending_updates=True)
        # run_polling returned cleanly — restart immediately
        logger.warning("Polling ended unexpectedly, restarting in 5s...")
        time.sleep(5)
    except Exception as exc:
        backoff = min(MAX_BACKOFF, 5 * attempt)
        logger.error(f"Bot crashed: {exc}. Restarting in {backoff}s...")
        time.sleep(backoff)
