"""
keep_alive.py — Flask keep-alive server for Render.com free tier.
Runs on port 8080 in a daemon thread so it doesn't block the Telegram bot.
"""

import logging
from threading import Thread
from flask import Flask

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def home():
    return "Bot is alive", 200


@app.route("/health")
def health():
    return "OK", 200


def _run():
    """Start Flask on 0.0.0.0 using Render's PORT env var (fallback 8080)."""
    import os
    port = int(os.environ.get("PORT", 8080))
    # Suppress noisy Flask/Werkzeug logs so they don't clutter bot output
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def keep_alive():
    """Launch the Flask server in a background daemon thread."""
    t = Thread(target=_run, daemon=True)
    t.start()
    logger.info("Keep-alive server started")
