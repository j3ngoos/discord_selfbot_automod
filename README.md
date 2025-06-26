![image](https://github.com/user-attachments/assets/69ebbdd0-e09b-4e86-bffb-343b24a7386b)
![image](https://github.com/user-attachments/assets/e10aa502-1c58-496b-9318-7a2bf524d5e0)


---

## 🔧 Шаг 1: Настройка файла `.env`

Открой файл `.env` (создай его, если его нет) в папке проекта и добавь в него следующие строки:

```
DISCORD_TOKEN=твой_токен_дискорд_аккаунта
WEBHOOK_URL=твой_вебхук_для_логов
```

Если ты используешь Discord-сервер **Lounge (lifehack)**, эти значения будут подходить по умолчанию. Иначе — подставь данные от своего сервера.

---

## 💻 Шаг 1.1: Установи Python 3.10.0

Скачай и установи нужную версию Python:

🔗 [Скачать Python 3.10.0](https://www.python.org/downloads/release/python-3100/)

Во время установки **обязательно отметь галочку “Add Python to PATH”**.

---

## 🧠 Шаг 1.2: Установи Ollama

Скачай Ollama для своей ОС:

🔗 [https://ollama.com/download](https://ollama.com/download)

Установи программу, следуя инструкциям. После установки перезагрузи компьютер (иногда требуется для работы).

---

## 🚀 Шаг 2: Запусти Mistral в Ollama

Открой **командную строку (cmd)** и выполни команду:

```bash
ollama run mistral
```

Подожди, пока модель полностью скачается (около **4 ГБ**, потребуется стабильный интернет). Когда увидишь сообщение `> mistral ready`, можно закрыть это окно.

---

## 📂 Шаг 2.1: Перейди в папку проекта

Открой **новое окно командной строки** и перейди в директорию, где находится твой бот:

```bash
cd путь\до\папки\с\ботом
```

Пример:

```bash
cd C:\Users\Имя\Desktop\discord-moderator
```

---

## 🐍 Шаг 2.2: Активируй виртуальное окружение

Если ты уже создал виртуальное окружение, активируй его:

```bash
myenv\Scripts\activate
```

Если не создавал, сделай это:

```bash
python -m venv myenv
myenv\Scripts\activate
```

Затем установи зависимости:

```bash
pip install -r requirements.txt
```

---

## ▶️ Шаг 2.3: Запусти бота

Запусти файл с ботом:

```bash
python discord_moderator_bot.py
```

Подожди, пока он скачает все нужные зависимости (например, токенизаторы модели).

---

## ⚙️ Шаг 3: Выбор режима модерации

После запуска бот предложит выбрать режим:

* `auto` — всё происходит **автоматически**, без подтверждений
* `hand` — каждый шаг **требует ручного подтверждения**

Просто введи `auto` или `hand` и нажми Enter.

---

## ✅ Шаг 4: Готово!

Теперь бот работает. Он будет:

* читать сообщения с серверов,
* анализировать их на токсичность,
* при необходимости отправлять в **Mistral** через **Ollama**,
* автоматически логировать действия в Discord через Webhook.

---

## 🔗 Полезные ссылки:

* 🔹 Discord.py-Self с поддержкой slash-команд: [https://github.com/dolfies/discord.py-self](https://github.com/dolfies/discord.py-self)
* 🔹 Mistral на Ollama: [https://ollama.com/library/mistral](https://ollama.com/library/mistral)
* 🔹 Как создать Webhook в Discord: [https://support.discord.com/hc/ru/articles/228383668](https://support.discord.com/hc/ru/articles/228383668)

---






---

## 🔧 Step 1: Create and Configure `.env` File

Open or create a `.env` file in your project folder and add your Discord token and webhook:

```
DISCORD_TOKEN=your_discord_token_here
WEBHOOK_URL=your_discord_webhook_here
```

> If you're using the **Lounge (lifehack)** server, default values should already match. Otherwise, insert your own server's credentials.

---

## 💻 Step 1.1: Install Python 3.10.0

Download and install Python 3.10.0 from the official site:

🔗 [Download Python 3.10.0](https://www.python.org/downloads/release/python-3100/)

✅ **Important:** During installation, check the box that says **“Add Python to PATH”**.

---

## 🧠 Step 1.2: Install Ollama

Download Ollama for your operating system:

🔗 [https://ollama.com/download](https://ollama.com/download)

Install it and **restart your PC** if prompted (some setups require this).

---

## 🚀 Step 2: Run Mistral via Ollama

Open **Command Prompt (cmd)** and type:

```bash
ollama run mistral
```

Wait for the Mistral model to fully download (\~4 GB). When you see `> mistral ready`, you can close that window.

---

## 📂 Step 2.1: Go to Your Project Folder

Open **a new cmd window** and navigate to the folder where your bot is located:

```bash
cd path\to\your\folder
```

Example:

```bash
cd C:\Users\YourName\Desktop\discord-moderator
```

---

## 🐍 Step 2.2: Activate Virtual Environment

If you already have a virtual environment:

```bash
myenv\Scripts\activate
```

If not, create one:

```bash
python -m venv myenv
myenv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Step 2.3: Run the Bot

Start the bot script:

```bash
python discord_moderator_bot.py
```

Wait while any remaining resources are downloaded (e.g., tokenizer files).

---

## ⚙️ Step 3: Choose Moderation Mode

When prompted:

* Type `auto` for **fully automatic mode** (no GUI confirmations)
* Type `hand` for **manual mode** (you approve each moderation step)

---

## ✅ Step 4: All Set!

The bot is now running. It will:

* monitor messages in selected channels,
* check toxicity using RuBERT,
* verify borderline cases with **Mistral via Ollama**,
* keep violation history,
* and send punishment logs to your Discord webhook.

---

## 🔗 Helpful Links

* 🔹 `discord.py-self` with slash command support: [https://github.com/dolfies/discord.py-self](https://github.com/dolfies/discord.py-self)
* 🔹 Mistral on Ollama: [https://ollama.com/library/mistral](https://ollama.com/library/mistral)
* 🔹 How to create a Discord Webhook: [https://support.discord.com/hc/en-us/articles/228383668](https://support.discord.com/hc/en-us/articles/228383668)





