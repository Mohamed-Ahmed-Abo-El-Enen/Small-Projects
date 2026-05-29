# OpenClaw on Google Colab - Complete Setup Guide

This guide will help you set up OpenClaw with Ollama models on Google Colab (free tier) and integrate it with Telegram for a powerful AI assistant experience.

## 🎯 What is This?

This notebook allows you to run OpenClaw (an AI coding assistant) on Google Colab's free GPU using Ollama models, with Telegram integration for remote access to your AI assistant.

## ✨ Features

- ✅ Free GPU-powered AI assistant on Google Colab
- ✅ Uses Ollama with GPT-OSS 20B model
- ✅ Telegram bot integration for remote access
- ✅ Web dashboard interface
- ✅ Public tunnel via Cloudflare for external access
- ✅ Full terminal access within Colab

## 📋 Prerequisites

1. **Google Account** - For Google Colab access
2. **Telegram Account** - For bot integration
3. **Telegram Bot Token** - Create one via [@BotFather](https://t.me/botfather)

### Creating a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the prompts to name your bot
4. Save the **Bot Token** provided (format: `110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw`)

## 🚀 Installation Steps

### Step 1: Setup Environment (Cells 1-6)

Run these cells in sequence to install dependencies:

**Cell 1: Install Node.js via NVM**
```bash
%%bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
npm install -g n
n lts
hash -r
node -v
source ~/.bashrc
```

**Cell 2: Install Cloudflared (for public tunneling)**
```bash
%%bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
mv cloudflared /usr/local/bin/cloudflared
```

**Cell 3: Configure Node.js version**
```bash
%%bash
source ~/.nvm/nvm.sh
nvm install 24
nvm use 24
nvm alias default 24
```

**Cell 4: Install Ollama**
```bash
%%bash
sudo apt install pciutils lshw zstd
curl -fsSL https://ollama.com/install.sh | sh
```

**Cell 5: Install OpenClaw and related tools**
```bash
%%bash
npm install -g openclaw@latest
npx skills add openclaw/skills
npm install -g molthub
```

**Cell 6: Install Python dependencies**
```python
!pip install colab-xterm
!pip install load_dotenv
```

### Step 2: Configure Environment Variables

**Cell 12: Set up your configuration**

Before running this cell, create a `.env` file or directly set your Telegram bot token:

```python
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["DEFAULT_NUM_CTX"] = "16384"
os.environ["TELEGRAM_BOT_TOKEN"] = "YOUR_TELEGRAM_BOT_TOKEN_HERE"

OLLAMA_MODEL = "gpt-oss:20b"
```

**Replace** `YOUR_TELEGRAM_BOT_TOKEN_HERE` with your actual Telegram bot token.

### Step 3: Start Ollama Server

**Cell 9: Start Ollama server**
```python
import subprocess
import time

OLLAMA_MODEL = "gpt-oss:20b"

print("Starting Ollama server...")
process = subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

time.sleep(10)
```

**Cell 10: Pull the AI model**
```python
print(f"Pulling model {OLLAMA_MODEL} from Ollama ...")
!ollama pull $OLLAMA_MODEL
```

⏱️ **Note:** Model download may take 5-15 minutes depending on connection speed (gpt-oss:20b is ~11GB).

### Step 4: Initial OpenClaw Configuration

**Cell 17: Open terminal for onboarding**

Run this cell to open an interactive terminal:

```python
%xterm
```

In the terminal that opens, run:

```bash
openclaw onboard --install-daemon
```

Follow the prompts:
1. Select `ollama/gpt-oss:20b` as your model
2. Complete the setup wizard
3. After completion, run: `source /root/.bashrc`

### Step 5: Telegram Integration

**Cell 18: Approve Telegram pairing**

1. Start a conversation with your Telegram bot
2. Send any message to get a **pairing code**
3. In a new terminal cell or in the existing xterm, run:

```bash
openclaw pairing approve telegram <YOUR_PAIRING_CODE>
```

Replace `<YOUR_PAIRING_CODE>` with the code from your Telegram bot.

### Step 6: Launch Web Dashboard

**Cell 21: Start the web dashboard**

```python
%xterm
```

In the terminal, run:

```bash
systemctl --user restart openclaw-gateway
openclaw dashboard --no-open
```

The dashboard will be available at `http://127.0.0.1:18789/`

### Step 7: Create Public Tunnel (Optional)

**Cell 22: Expose via Cloudflare tunnel**

To access your dashboard from anywhere:

```python
!cloudflared tunnel --url http://127.0.0.1:18789/?token=YOUR_TOKEN_HERE
```

This will generate a public URL (like `https://random-name.trycloudflare.com`) that you can access from any device.

## 🎮 Usage

### Via Telegram

1. Open your Telegram bot
2. Send messages to interact with your AI assistant
3. OpenClaw will process your requests using the Ollama model

### Via Web Dashboard

1. Access the local URL or public Cloudflare URL
2. Use the web interface to interact with OpenClaw
3. Monitor sessions and configurations

### Via Terminal

Use the `%xterm` cells to access the terminal directly and run OpenClaw commands:

```bash
# Check status
openclaw status

# List sessions
openclaw sessions list

# Run a task
openclaw run "your task here"
```

## 🔧 Useful Commands

### Configuration Management

```python
# View current configuration
import json
import os
from IPython.display import JSON

file_path = os.path.expanduser('~/.openclaw/openclaw.json')
with open(file_path, 'r') as f:
    data = json.load(f)
JSON(data)
```

### Reset Configuration

```python
import os
import shutil

def reset_all_config():
    paths = [
        '~/.openclaw/openclaw.json',
        '~/.openclaw/credentials',
        '~/.openclaw/agents/main/sessions',
        '~/.openclaw/workspace'
    ]
    for path in paths:
        full_path = os.path.expanduser(path)
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
        except FileNotFoundError:
            pass

# reset_all_config()  # Uncomment to use
```

### Check Port Status

```bash
!ss -tulpn | grep 18789
```

### Kill OpenClaw Process

```bash
!pkill -f openclaw
```

## ⚠️ Important Notes

### Google Colab Limitations

1. **Session Timeout**: Colab sessions disconnect after ~12 hours or 90 minutes of inactivity
2. **GPU Availability**: Free tier has limited GPU hours per week
3. **Storage**: Files are ephemeral; they're deleted when the runtime disconnects
4. **Network**: Some ports may be restricted

### Best Practices

1. **Save your work**: Download important configurations before disconnecting
2. **Model persistence**: You'll need to re-download the model if runtime resets
3. **Environment variables**: Set your Telegram token at the start of each session
4. **Monitoring**: Keep an eye on GPU usage and memory

### Troubleshooting

**Ollama server won't start:**
- Restart the runtime (Runtime → Restart runtime)
- Re-run installation cells

**Model download fails:**
- Check internet connection
- Try a smaller model: `qwen2.5:7b` or `llama3.2:3b`

**Telegram bot not responding:**
- Verify bot token is correct
- Check pairing code was approved
- Restart OpenClaw gateway

**Web dashboard not accessible:**
- Ensure port 18789 is not blocked
- Try restarting the gateway: `systemctl --user restart openclaw-gateway`

**Out of memory errors:**
- Use a smaller model
- Reduce `DEFAULT_NUM_CTX` value
- Restart runtime to clear memory

## 📚 Additional Resources

- [OpenClaw Documentation](https://github.com/openclaw/openclaw)
- [Ollama Models](https://ollama.com/library)
- [Telegram Bot API](https://core.telegram.org/bots)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

## 🤝 Alternative Models

You can use different models by changing the `OLLAMA_MODEL` variable:

- `qwen2.5:7b` - Lighter, faster (4GB)
- `llama3.2:3b` - Very light (2GB)
- `mistral:7b` - Good balance (4GB)
- `qwen3-vl:8b` - Vision-capable model (5GB)

To change model:
```python
OLLAMA_MODEL = "qwen2.5:7b"  # Change this
!ollama pull $OLLAMA_MODEL
```

## 🎉 Quick Start Summary

1. Run cells 1-6 (installation)
2. Set Telegram token in cell 12
3. Run cells 9-10 (start Ollama + download model)
4. Run cell 17, execute `openclaw onboard --install-daemon`
5. Pair Telegram bot (cell 18 area)
6. Launch dashboard (cell 21)
7. Create public tunnel if needed (cell 22)
8. Start chatting with your AI assistant!

## 📝 License

This guide is provided as-is for educational purposes. Refer to OpenClaw and Ollama licensing for commercial use.

---

**Happy AI Coding! 🚀**

For issues or questions, check the OpenClaw GitHub repository or community forums.