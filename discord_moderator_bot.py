import discord
import aiohttp
import asyncio
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from collections import defaultdict
import re
import threading
import pygame
import json
from tkinter import messagebox, Tk

# --- Загрузка ENV ---
load_dotenv()
TOKEN = os.getenv("TOKEN")
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", 0.7))
CHANNELS_FORCHECK = [int(x) for x in os.getenv("CHANNELS_Forcheck", "").split(",") if x]
MUTE_BOT_ID = int(os.getenv("MUTE_BOT_ID"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

IGNORED_USERS = [709716833841971231]  # замените на реальные user_id
MODERATOR_ID = 848876544536608768
MANUAL_MODE = True

pygame.mixer.init()

VIOLATIONS_PATH = "violations.json"
def load_violations():
    if os.path.exists(VIOLATIONS_PATH):
        with open(VIOLATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_violations(data):
    with open(VIOLATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

violations_data = load_violations()

def add_violation(uid):
    now_str = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    entry = violations_data.get(str(uid), {"total": 0, "history": [], "last_violation": None})
    entry["total"] += 1
    entry["last_violation"] = now_str
    entry["history"].append("2.2")
    violations_data[str(uid)] = entry
    save_violations(violations_data)

def play_sound(path="sound.mp3"):
    def _play():
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"[SOUND ERROR] {e}")
    threading.Thread(target=_play, daemon=True).start()

def gui_confirm(user_id, username, text, action):
    play_sound("sound.mp3")
    root = Tk()
    root.withdraw()
    ok = messagebox.askokcancel(
        "Подтвердить действие",
        f"Пользователь: {username} ({user_id})\n"
        f"Сообщение:\n{text}\n\n"
        f"Действие: {action}\n\nПодтвердить?"
    )
    root.destroy()
    return ok

def get_toxicity_score(text):
    tokens = rubert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = rubert_model(**tokens).logits
        probs = F.softmax(logits, dim=1)
    return probs[0][1].item()

async def check_mistral(text):
    prompt = f"""Ты — помощник модератора Discord. Твоя задача — определить, нарушает ли приведённое ниже сообщение правило 2.2 сервера.

🔹 **Правило 2.2:**
Запрещены многочисленные оскорбления других пользователей или целенаправленные провокации. Просто использование мата или эмоциональные выражения (например: "блин", "жесть", "это пиздец") — не считаются нарушением, если не направлены против конкретных людей.

Примеры нарушений:
- "Ты дебил, идиот, заткнись"
- "Вы твари, как с вами можно говорить?"
- "Закрой рот, животное"

Примеры допустимых выражений:
- "это пиздец"
- "ну и жопа"
- "как же меня всё бесит!"

🔹 **Сообщение:**
 "{text.strip()}" 

Ответь только на **русском языке**, строго одним из двух вариантов:

- **"1"** - нарушение 
- **"0"** - нет нарушения
"""
    payload = {
        "model": MISTRAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
            async with sess.post(OLLAMA_URL, json=payload) as resp:
                data = await resp.json()
                out = data.get("response", "").strip()
                print(f"[MISTRAL RAW]: {out}")
                clean = re.sub(r'[\*\s"«»-]+', '', out)
                return clean.startswith("1")
    except Exception as e:
        print(f"[MISTRAL ERROR] {e}")
        return False

async def log_action(channel_id, user_id, username, ts, text, rule, action):
    if not WEBHOOK_URL:
        return
    content = (
        f"🔔 Нарушение\n"
        f"Канал: <#{channel_id}>\n"
        f"Автор: <@{user_id}> ({username})\n"
        f"Время: {ts}\n"
        f"Сообщение: \"{text}\"\n"
        f"Правило: {rule}\n"
        f"Действие: {action}"
    )
    async with aiohttp.ClientSession() as sess:
        await sess.post(WEBHOOK_URL, json={"content": content})

def is_adekvatnee(text):
    return bool(re.search(r"адек.*?тн.*?", text, re.IGNORECASE | re.DOTALL))

def is_pred_22(text):
    return bool(re.search(r"\\bпред[^\n]*2\\.2", text, re.IGNORECASE))

class ModeratorClient(discord.Client):
    async def on_ready(self):
        print(f"[LOGIN] {self.user} ({self.user.id})")

    async def on_message(self, message):
        now = datetime.now(timezone(timedelta(hours=3)))
        if now.hour not in [9, 13, 17]: return
        if message.author.id == self.user.id: return

        # --- Чтение от модератора ---
        if message.author.id == MODERATOR_ID:
            if message.mentions:
                mentioned_id = message.mentions[0].id
                if is_adekvatnee(message.content) or is_pred_22(message.content):
                    print(f"[MODLOG] Нарушение от модера: {mentioned_id}")
                    add_violation(mentioned_id)
            return

        if message.channel.id not in CHANNELS_FORCHECK: return

        txt = message.content
        uid = message.author.id
        if uid in IGNORED_USERS:
            return
        uname = str(message.author)
        cid = message.channel.id

        score = get_toxicity_score(txt)
        print(f"[TOX] {score:.3f} | {uname} | {txt}")
        if score < TOXICITY_THRESHOLD:
            return

        if not await check_mistral(txt):
            print("[MISTRAL] Нет нарушения")
            return

        user_violations[uid] += 1
        cnt = user_violations[uid]
        if uid in violation_timers:
            violation_timers[uid].cancel()
        violation_timers[uid] = asyncio.get_event_loop().call_later(
            600, lambda: user_violations.pop(uid, None)
        )
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        if cnt == 1:
            action = f"<@{uid}> — адекватнее"
        elif cnt == 2:
            action = f"<@{uid}> — пред 2.2"
        elif cnt == 3:
            cmds = await message.channel.application_commands()
            mute = [c for c in cmds if c.name=="mute" and str(c.application_id)==str(MUTE_BOT_ID)]
            if mute:
                await mute[0].__call__(message.channel, user=uid, time="20", reason="2.2")
            else:
                action = "mute команда не найдена"
            user_violations.pop(uid, None)
            violation_timers.pop(uid, None)
        else:
            return

        ok = True
        if MANUAL_MODE:
            ok = gui_confirm(uid, uname, txt, action)
        if not ok:
            user_violations[uid] = max(user_violations.get(uid,1)-1,0)
            print("[INFO] Нарушение отменено")
            return

        try:
            await message.delete()
            print("[DELETE]", txt)
        except Exception as e:
            print("[WARN] Не удалось удалить:", e)

        await message.channel.send(action)
        print("[ACTION]", action)

        add_violation(uid)
        await log_action(cid, uid, uname, ts, txt, "2.2", action)

rubert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
rubert_model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny-toxicity"
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
rubert_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_violations = defaultdict(int)
violation_timers = {}

if __name__ == "__main__":
    mode = input("Режим работы — авто (a) / ручной (m)? [a/m]: ").strip().lower()
    MANUAL_MODE = not mode.startswith("a")
    print("Ручной режим:", MANUAL_MODE)
    client = ModeratorClient()
    asyncio.run(client.start(TOKEN))
