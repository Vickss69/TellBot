"""
bot.py — Telegram chatbot powered by Hugging Face LLM models.
Main entry point. Uses polling mode (no webhooks).
"""

import os
import logging
import asyncio
from typing import Any

from huggingface_hub import InferenceClient
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction

from keep_alive import keep_alive

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Crash loudly if tokens are missing
TG_TOKEN: str = os.environ["TG_TOKEN"]
HF_TOKEN: str = os.environ["HF_TOKEN"]

# Logging with timestamps
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MODELS REGISTRY
# ─────────────────────────────────────────────

MODELS: list[dict[str, str]] = [
    {"display": "⚡ Zephyr 7B", "id": "HuggingFaceH4/zephyr-7b-beta"},
    {"display": "🌀 Mistral 7B", "id": "mistralai/Mistral-7B-Instruct-v0.3"},
    {"display": "🔮 Qwen 2.5 7B", "id": "Qwen/Qwen2.5-7B-Instruct"},
    {"display": "📚 Flan-T5 Large", "id": "google/flan-t5-large"},
    {"display": "🦙 Llama 3.1 8B", "id": "meta-llama/Llama-3.1-8B-Instruct"},
    {"display": "🌙 Kimi K2.5", "id": "moonshotai/Kimi-K2.5"},
]

DEFAULT_MODEL_ID: str = "HuggingFaceH4/zephyr-7b-beta"

SYSTEM_PROMPT: str = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Be concise but thorough."
)

MAX_CONTEXT_MESSAGES: int = 20  # 10 user + 10 assistant
TELEGRAM_MSG_LIMIT: int = 4096

# ─────────────────────────────────────────────
# PER-USER STATE  (in-memory dict)
# ─────────────────────────────────────────────

# {user_id: {"model_id": str, "history": [{"role": ..., "content": ...}, ...]}}
user_state: dict[int, dict[str, Any]] = {}


def get_state(user_id: int) -> dict[str, Any]:
    """Return (or create) the state dict for a given user."""
    if user_id not in user_state:
        user_state[user_id] = {
            "model_id": DEFAULT_MODEL_ID,
            "history": [],
        }
    return user_state[user_id]


def display_name_for(model_id: str) -> str:
    """Look up the friendly display name for a model id."""
    for m in MODELS:
        if m["id"] == model_id:
            return m["display"]
    return model_id


# ─────────────────────────────────────────────
# HF API CALL LOGIC
# ─────────────────────────────────────────────

def format_prompt_manually(history: list[dict[str, str]]) -> str:
    """Build a plain-text prompt from the conversation history for models
    that don't support chat_completion."""
    lines: list[str] = []
    lines.append(f"System: {SYSTEM_PROMPT}\n")
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def handle_error(exc: Exception) -> str:
    """Convert an HF API exception into a user-friendly message."""
    err = str(exc).lower()
    status = ""
    # Try to extract HTTP status from the error string
    for code in ("503", "429", "401", "403", "422", "404"):
        if code in str(exc):
            status = code
            break

    if status == "503" or "loading" in err or "currently loading" in err:
        return (
            "⏳ The model is loading (cold start). "
            "Please wait 30–60 seconds and try again."
        )
    if status == "429" or "rate" in err:
        return "⚠️ Rate limited by Hugging Face. Please wait a minute and retry."
    if status == "410" or "no longer supported" in err or "gone" in err:
        return (
            "🚫 This model is no longer available on Hugging Face Inference API. "
            "Use /model to pick another model."
        )
    if (
        status in ("401", "403")
        or "gated" in err
        or "access" in err
        or "authorization" in err
        or "restricted" in err
        or "forbidden" in err
    ):
        return (
            "🔒 This model requires special access / approval on Hugging Face. "
            "Use /model to pick another model."
        )
    if "timed out" in err or "timeout" in err:
        return (
            "⏰ Request timed out. The model might be overloaded. "
            "Please try again in a moment."
        )
    # Generic fallback — cap at 300 chars
    brief = str(exc)[:300]
    return f"❌ Error: {brief}"


def ask_model(model_id: str, history: list[dict[str, str]]) -> str:
    """Send conversation to HF model and return the assistant reply text."""
    client = InferenceClient(token=HF_TOKEN)
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    # ── Attempt 1: chat_completion ──
    try:
        response = client.chat_completion(
            model=model_id,
            messages=[system_msg] + history,
            max_tokens=1024,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
        return ""
    except Exception as e:
        err_str = str(e).lower()
        needs_fallback = (
            "not supported" in err_str
            or "422" in str(e)
            or "not available" in err_str
            or "does not support" in err_str
            or "unprocessable" in err_str
        )
        if needs_fallback:
            # ── Attempt 2: text_generation fallback ──
            try:
                prompt = format_prompt_manually(history)
                result = client.text_generation(
                    model=model_id,
                    prompt=prompt,
                    max_new_tokens=1024,
                    temperature=0.7,
                )
                if result:
                    return result.strip()
                return ""
            except Exception as e2:
                return handle_error(e2)
        else:
            return handle_error(e)


# ─────────────────────────────────────────────
# TELEGRAM HANDLERS
# ─────────────────────────────────────────────

WELCOME_TEXT = (
    "🤖 <b>Welcome to the AI Chat Bot!</b>\n\n"
    "I connect you to powerful open-source LLMs hosted on Hugging Face.\n\n"
    "<b>Available commands:</b>\n"
    "/model  — Choose an AI model\n"
    "/current — Show your active model & context size\n"
    "/clear  — Clear your conversation history\n"
    "/help   — Show this help message\n\n"
    "Just send me any text message and I'll reply using the selected model.\n\n"
    "💡 <i>Default model: ⚡ Zephyr 7B</i>"
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /help."""
    _ = get_state(update.effective_user.id)
    await update.message.reply_text(WELCOME_TEXT, parse_mode="HTML")


async def cmd_current(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /current — show active model + context count."""
    state = get_state(update.effective_user.id)
    name = display_name_for(state["model_id"])
    count = len(state["history"])
    text = (
        f"📊 <b>Current session info</b>\n\n"
        f"<b>Model:</b> {name}\n"
        f"<b>Model ID:</b> <code>{state['model_id']}</code>\n"
        f"<b>Messages in context:</b> {count}"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear — wipe conversation history."""
    state = get_state(update.effective_user.id)
    state["history"] = []
    await update.message.reply_text(
        "🗑️ Conversation history cleared. Send a message to start fresh!"
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model and /models — show inline keyboard of model choices."""
    state = get_state(update.effective_user.id)
    current_id = state["model_id"]

    buttons: list[list[InlineKeyboardButton]] = []
    for m in MODELS:
        prefix = "✅ " if m["id"] == current_id else ""
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{prefix}{m['display']}",
                    callback_data=f"select_model:{m['id']}",
                )
            ]
        )

    markup = InlineKeyboardMarkup(buttons)
    await update.message.reply_text(
        "🔧 <b>Select a model:</b>",
        reply_markup=markup,
        parse_mode="HTML",
    )


async def callback_model_select(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle inline keyboard button press for model selection."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data or not data.startswith("select_model:"):
        return

    model_id = data.split(":", 1)[1]
    user_id = query.from_user.id
    state = get_state(user_id)

    if state["model_id"] == model_id:
        await query.edit_message_text(
            f"ℹ️ You're already using <b>{display_name_for(model_id)}</b>.",
            parse_mode="HTML",
        )
        return

    old_name = display_name_for(state["model_id"])
    state["model_id"] = model_id
    state["history"] = []  # clear history on switch
    new_name = display_name_for(model_id)

    await query.edit_message_text(
        f"✅ Switched from <b>{old_name}</b> → <b>{new_name}</b>\n"
        f"<code>{model_id}</code>\n\n"
        "🗑️ Conversation history cleared. Send a message to start chatting!",
        parse_mode="HTML",
    )
    logger.info("User %s switched model to %s", user_id, model_id)


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle any non-command text message — send to HF model."""
    user_id = update.effective_user.id
    state = get_state(user_id)
    user_text = update.message.text

    if not user_text or not user_text.strip():
        return

    # Append user message to history
    state["history"].append({"role": "user", "content": user_text})

    # Trim history to last MAX_CONTEXT_MESSAGES
    if len(state["history"]) > MAX_CONTEXT_MESSAGES:
        state["history"] = state["history"][-MAX_CONTEXT_MESSAGES:]

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    # Call the HF model in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    try:
        reply = await loop.run_in_executor(
            None, ask_model, state["model_id"], list(state["history"])
        )
    except Exception as exc:
        logger.exception("Unexpected error calling ask_model for user %s", user_id)
        reply = handle_error(exc)

    # Handle empty response
    if not reply or not reply.strip():
        reply = (
            "🤔 The model returned an empty response. "
            "Try rephrasing or switch models with /model"
        )

    # Append assistant reply to history
    state["history"].append({"role": "assistant", "content": reply})

    # Trim again after appending
    if len(state["history"]) > MAX_CONTEXT_MESSAGES:
        state["history"] = state["history"][-MAX_CONTEXT_MESSAGES:]

    # Send reply, splitting if too long for Telegram
    await send_long_message(update, reply)


async def send_long_message(update: Update, text: str) -> None:
    """Send a message, splitting into chunks if it exceeds Telegram's limit."""
    if len(text) <= TELEGRAM_MSG_LIMIT:
        await update.message.reply_text(text)
        return

    # Split into chunks at the last newline before the limit, fallback to hard split
    chunks: list[str] = []
    while text:
        if len(text) <= TELEGRAM_MSG_LIMIT:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, TELEGRAM_MSG_LIMIT)
        if split_at == -1 or split_at < TELEGRAM_MSG_LIMIT // 2:
            # Try space
            split_at = text.rfind(" ", 0, TELEGRAM_MSG_LIMIT)
        if split_at == -1 or split_at < TELEGRAM_MSG_LIMIT // 2:
            split_at = TELEGRAM_MSG_LIMIT
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    for i, chunk in enumerate(chunks):
        if chunk.strip():
            await update.message.reply_text(chunk)


async def handle_unknown(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "❓ Unknown command. Use /help to see available commands."
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """Start the bot."""
    # Start keep-alive Flask server
    keep_alive()

    logger.info("Starting Telegram bot...")

    app = (
        Application.builder()
        .token(TG_TOKEN)
        .build()
    )

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("models", cmd_model))
    app.add_handler(CommandHandler("current", cmd_current))
    app.add_handler(CommandHandler("clear", cmd_clear))

    # Callback query handler (inline keyboard)
    app.add_handler(CallbackQueryHandler(callback_model_select, pattern=r"^select_model:"))

    # Message handler (non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Unknown command handler (must be last)
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown))

    logger.info("Bot is ready. Polling for updates...")

    # Run with polling, drop pending updates so we don't replay old messages
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
