"""
Telegram bot for S&P 500 trading signals.

Commands / buttons:
  /start        — welcome + main menu
  /analyze      — on-demand S&P 500 analysis
  /status       — quick price + signal summary
  /alerts on|off — toggle proactive alerts for this chat
  /help         — command list

Proactive alerts: background job runs every 15 minutes and notifies
all opted-in chats when a big buying opportunity is detected.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

import signals as spy

logger = logging.getLogger("bot")

# ── Chat registry for proactive alerts ───────────────────────────────────────
_STATE_FILE = Path("spy_state.json")
ALERT_INTERVAL_MINUTES = 15


def _load_chats() -> set[int]:
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text())
            chats = set(data.get("alert_chats", []))
            logger.info(f"Loaded {len(chats)} subscriber(s)")
            return chats
    except Exception as e:
        logger.warning(f"Could not load state: {e}")
    return set()


def _save_chats(chats: set[int]) -> None:
    try:
        _STATE_FILE.write_text(json.dumps({"alert_chats": list(chats)}))
    except Exception as e:
        logger.warning(f"Could not save state: {e}")


_alert_chats: set[int] = _load_chats()


# ── Keyboards ─────────────────────────────────────────────────────────────────

def _main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Analyze Market", callback_data="analyze"),
            InlineKeyboardButton("⚡ Quick Status",   callback_data="status"),
        ],
        [
            InlineKeyboardButton("🔔 Enable Alerts",  callback_data="alerts_on"),
            InlineKeyboardButton("🔕 Disable Alerts", callback_data="alerts_off"),
        ],
    ])


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _alert_chats.add(chat_id)
    _save_chats(_alert_chats)
    await update.message.reply_text(
        "👋 *S&P 500 Signal Bot*\n\n"
        "I analyze the S&P 500 (SPY) using RSI, MACD, Bollinger Bands, "
        "moving averages, and volume to give you real-time trading signals.\n\n"
        "🔔 *Proactive alerts are ON* — I'll ping you whenever I spot a "
        "strong buying opportunity (checks every 15 min).\n\n"
        "Tap a button below to get started:",
        reply_markup=_main_menu(),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*S&P 500 Signal Bot — Commands*\n\n"
        "/start — main menu\n"
        "/analyze — full market analysis\n"
        "/status — quick price snapshot\n"
        "/alerts on — enable proactive alerts\n"
        "/alerts off — disable proactive alerts\n"
        "/help — this message",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _do_analyze(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, *, is_quick: bool = False) -> None:
    """Run signal analysis and send result to chat_id."""
    msg = await ctx.bot.send_message(chat_id, "⏳ Analyzing the market…")
    try:
        sig = await asyncio.get_event_loop().run_in_executor(None, spy.analyze)
        text = spy.format_signal_message(sig)
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_menu(),
        )
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"⚠️ Analysis failed: {exc}\n\nMarket data may be unavailable outside US trading hours.",
            reply_markup=_main_menu(),
        )


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _do_analyze(update.effective_chat.id, ctx)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("⏳ Fetching…")
    try:
        sig = await asyncio.get_event_loop().run_in_executor(None, spy.analyze)
        emoji = {"STRONG BUY": "🟢", "BUY": "📈", "HOLD": "⚖️", "SELL": "📉", "STRONG SELL": "🔴"}.get(sig.direction, "❓")
        chg = f"+{sig.change_1d:.2f}%" if sig.change_1d >= 0 else f"{sig.change_1d:.2f}%"
        text = (
            f"{emoji} *{sig.direction}*  ({sig.confidence*100:.0f}%)\n"
            f"SPY ${sig.price:.2f}  {chg} today\n"
            f"RSI {sig.rsi:.1f} · MACD {sig.macd_signal}"
        )
        await ctx.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=msg.message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_menu(),
        )
    except Exception as exc:
        await ctx.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=msg.message_id,
            text=f"⚠️ Failed: {exc}",
        )


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    args = ctx.args or []
    if args and args[0].lower() == "off":
        _alert_chats.discard(chat_id)
        _save_chats(_alert_chats)
        await update.message.reply_text("🔕 Proactive alerts *disabled* for this chat.", parse_mode=ParseMode.MARKDOWN)
    else:
        _alert_chats.add(chat_id)
        _save_chats(_alert_chats)
        await update.message.reply_text(
            "🔔 Proactive alerts *enabled*. I'll notify you when I spot a big buying opportunity.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data

    if data == "analyze":
        await _do_analyze(chat_id, ctx)
    elif data == "status":
        try:
            sig = await asyncio.get_event_loop().run_in_executor(None, spy.analyze)
            emoji = {"STRONG BUY": "🟢", "BUY": "📈", "HOLD": "⚖️", "SELL": "📉", "STRONG SELL": "🔴"}.get(sig.direction, "❓")
            chg = f"+{sig.change_1d:.2f}%" if sig.change_1d >= 0 else f"{sig.change_1d:.2f}%"
            text = (
                f"{emoji} *{sig.direction}*  ({sig.confidence*100:.0f}%)\n"
                f"SPY ${sig.price:.2f}  {chg} today\n"
                f"RSI {sig.rsi:.1f} · MACD {sig.macd_signal}"
            )
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=_main_menu())
        except Exception as exc:
            await query.edit_message_text(f"⚠️ Failed: {exc}", reply_markup=_main_menu())
    elif data == "alerts_on":
        _alert_chats.add(chat_id)
        _save_chats(_alert_chats)
        await query.answer("🔔 Alerts enabled!", show_alert=True)
    elif data == "alerts_off":
        _alert_chats.discard(chat_id)
        _save_chats(_alert_chats)
        await query.answer("🔕 Alerts disabled.", show_alert=True)


# ── Proactive alert job ───────────────────────────────────────────────────────

# Track buy and sell scores independently so one doesn't suppress the other
_last_buy_alert_score: float = 50.0
_last_sell_alert_score: float = 50.0


async def _alert_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global _last_buy_alert_score, _last_sell_alert_score
    if not _alert_chats:
        return

    try:
        sig = await asyncio.get_event_loop().run_in_executor(None, spy.analyze)
    except Exception as exc:
        logger.warning(f"Alert job analysis failed: {exc}")
        return

    if not sig.is_alert_worthy:
        return

    # Don't spam the same alert — only fire if score meaningfully changed
    if sig.alert_type == "sell":
        if abs(sig.score - _last_sell_alert_score) < 8:
            return
        _last_sell_alert_score = sig.score
    else:
        if abs(sig.score - _last_buy_alert_score) < 8:
            return
        _last_buy_alert_score = sig.score

    text = spy.format_alert_message(sig)
    for chat_id in list(_alert_chats):
        try:
            await ctx.bot.send_message(
                chat_id,
                text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=_main_menu(),
            )
            logger.info(f"Alert sent to chat {chat_id}: score={sig.score:.1f}")
        except Exception as exc:
            logger.warning(f"Failed to send alert to {chat_id}: {exc}")


async def _summary_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a market snapshot every 4 hours regardless of signal strength."""
    if not _alert_chats:
        return
    try:
        sig = await asyncio.get_event_loop().run_in_executor(None, spy.analyze)
        text = spy.format_signal_message(sig, header="🕐 Scheduled Market Update")
        for chat_id in list(_alert_chats):
            try:
                await ctx.bot.send_message(
                    chat_id, text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=_main_menu(),
                )
            except Exception as exc:
                logger.warning(f"Failed to send summary to {chat_id}: {exc}")
        logger.info(f"Scheduled summary sent to {len(_alert_chats)} chat(s)")
    except Exception as exc:
        logger.warning(f"Summary job failed: {exc}")


# ── App setup ─────────────────────────────────────────────────────────────────

def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("alerts",  cmd_alerts))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Proactive alert job — every 15 minutes
    app.job_queue.run_repeating(
        _alert_job,
        interval=ALERT_INTERVAL_MINUTES * 60,
        first=60,
    )

    # Scheduled market summary — every 4 hours
    app.job_queue.run_repeating(
        _summary_job,
        interval=4 * 60 * 60,
        first=30,   # send first summary 30s after startup
    )

    return app
