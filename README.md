<!-- Demo GIF inserted -->
### Demo Video 🎥  

![MedCalc-Agent Demo](static/demo.gif)

[Download the original MP4](video3770828466.mp4)

# MedCalc-Agent – Browser-Augmented Medical Calculator Assistant

MedCalc-Agent is a full-stack, browser-augmented LLM application that automatically extracts structured values from free-text clinical notes, feeds them into the appropriate calculator on **MDCalc.com**, and returns a neatly formatted answer—complete with a screenshot of the calculator result and a justification of every input value.  
It is built on **FastAPI** (Python) for the backend, **Playwright** for headless browser automation, and a lightweight HTML/JS frontend that communicates over WebSockets for real-time chat.

## Key Features

* 🔎 **Three-agent workflow** – one agent scrapes calculator field metadata, the second pulls values from patient notes, and the third fills the online form and captures the result.
* 🌐 **Browser automation** – Playwright drives Chromium to interact with MDCalc exactly as a clinician would, ensuring authentic calculations and full auditability.
* 💬 **Real-time chat UI** – a modern single-page interface (`index.html`, `static/js/app.js`, `static/css/style.css`) lets users select a calculator, paste a patient note, and converse with the assistant.
* 🤖 **Multiple model configurations** – easily switch between GPT-4o, GPT-4o-mini, or any OpenAI-compatible model, with optional vision support.
* 📈 **Result validation & analytics** – scripts such as `count_within_bounds.py` and `visualize_results.py` generate CSV summaries and dashboards for large-scale test runs.
* 💾 **Conversation persistence** – save and reload past sessions; everything is version-stamped and stored in `/saved_chats`.

## Project Layout (selected files)

```
app.py                       # FastAPI entry-point
routes/
    api.py                  # REST endpoints (chat, calculator selection, etc.)
    websocket.py            # WebSocket endpoint for live chat
static/
    js/app.js               # Frontend logic (WebSocket client, UI handlers)
    css/style.css           # Styling
    img/                    # Logos & icons
templates/index.html         # Main HTML template
browser_calculator.py        # Orchestrates Playwright + agents
workflow_agent.py            # Three-agent reasoning pipeline
run_csv_instances.py         # Batch driver for CSV test cases
visualize_results.py         # Generates dashboards from batch runs
```

## Quick Start

1. **Install dependencies**
```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
2. **Create a `.env`** with your OpenAI (or Azure OpenAI) creds:
   ```bash
   OPENAI_API_KEY=sk-...
   # Optional Azure vars
   # AZURE_OPENAI_API_KEY=
   # AZURE_ENDPOINT=
   ```
3. **Launch the server**
   ```bash
   python app.py
   ```
4. **Open** `http://localhost:8000` in your browser, choose a calculator, paste a clinical note, and hit **Send**.

A convenience script `run_chat_app.sh` does the same in one step (installs deps then starts the server).

## 🚀 Live Chat UI

Run the agent in your browser and chat with it in real time.

```bash
python app.py   # or ./run_chat_app.sh
```

Then open **http://localhost:8000** and you will see:

* A sidebar where you pick a calculator, change the OpenAI model, or start/save conversations.
* A chat pane that streams messages from the LLM and renders MD-style screenshots inline.

### Workflow

1. **Choose a calculator** from the dropdown.
2. Paste or type a **patient note**. The assistant extracts values, fills MDCalc via Playwright and returns the score + screenshot.
3. If you realise a value is wrong, just send a correction (e.g. "Weight is 80 kg, not 72 kg"). The browser stays open, the agent patches only that field, waits ~3 s for MDCalc to auto-refresh, grabs a new screenshot, and replies with the updated result.
4. Click **Save Conversation** in the sidebar at any point. A JSON file is written to `saved_chats/` (messages, calculator, URL, model).
5. Under *Saved Conversations* click any entry to reload it instantly—messages re-render, the calculator dropdown is reset, and you can resume chatting where you left off.

### Shortcuts & Tips

* `New Conversation` resets the chat but keeps the browser session alive.
* The model selector supports any OpenAI-compatible model name in your `.env`.
* Screenshots are written to `static/screenshots/` and referenced in chat via Markdown—so they load even after a restart.

## Batch Mode & Evaluation

To run hundreds of notes automatically:
```bash
python run_csv_instances.py --csv inputs.csv --model gpt-4o --vision yes
```
This produces a `results/…csv` file.  Visualise with:
```bash
python visualize_results.py results/summary_YYYYMMDD_HHMMSS.csv
```
An HTML dashboard and PNG charts will appear in `/visualizations`.

## Development Notes
* Playwright is launched with extra flags (`--disable-window-activation`, `--disable-focus-on-load`, retries, custom wait-states) to minimise flaky crashes.
* The UI reconnects its WebSocket automatically if the backend restarts, so you can develop without losing state.
* Calculator mappings live in `calculator_map.json`; no fragile string concatenation.

---
