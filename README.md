# Novel Chapter Generator

Generates new web novel chapters from an existing `.epub` or `.txt` file using:
- **DeepSeek V3** via OpenRouter (generation + planning)
- **LlamaIndex + ChromaDB** (local RAG vector store, persists between runs)
- **BAAI/bge-large-en-v1.5** (local embeddings, no API key needed)
- **ebooklib + lxml + BeautifulSoup4** (EPUB read/write with guaranteed Unicode fidelity)

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10 or newer
- An [OpenRouter](https://openrouter.ai) API key with DeepSeek V3 access

Check your Python version:
```bash
python3 --version
```

If you need to install Python, the easiest way on Mac is via [Homebrew](https://brew.sh):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

---

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

You will need to run `source venv/bin/activate` again any time you open a new terminal window.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

The first install will take a few minutes. The embedding model (~1.3 GB) downloads automatically on first run.

### 3. Set your OpenRouter API key
```bash
export OPENROUTER_API_KEY=sk-or-xxxx
```

To avoid setting this every session, add it to your shell profile:
```bash
echo 'export OPENROUTER_API_KEY=sk-or-xxxx' >> ~/.zshrc
source ~/.zshrc
```

---

## Generating Chapters

### Basic usage
```bash
python3 novel_generator.py my_novel.epub 3
```

### With a custom book ID (recommended)
```bash
python3 novel_generator.py my_novel.epub 5 --book-id cultivation_story
```

Using `--book-id` lets you give the book a memorable name and pick up where you left off in future sessions.

### Continue generating more chapters for the same book
```bash
python3 novel_generator.py my_novel.epub 5 --book-id cultivation_story
```
Run the same command again — the script detects the existing index and resumes from the last generated chapter automatically.

### Pass the API key inline instead of via env var
```bash
python3 novel_generator.py my_novel.epub 3 --api-key sk-or-xxxx
```

### TXT input is also supported
```bash
python3 novel_generator.py my_novel.txt 3 --book-id cultivation_story
```

---

## Input Format

### EPUB (recommended)
Any standard `.epub` file. The script reads chapters in spine order, skips short navigation pages, and detects chapter headers automatically. The output is a new EPUB file with the generated chapters appended — the original is never modified.

### TXT
The `.txt` file should have chapter headers in any common format:
```
Chapter 1: The Beginning
...chapter text...

Chapter 2 - The Journey
...chapter text...

CHAPTER 3
...chapter text...
```

---

## Output

### EPUB output (when input is EPUB)
```
novel_books/
  <book-id>/
    <original_name>_with_generated.epub    ← your original book + new chapters
```

Open this in any EPUB reader (Apple Books, Calibre, etc.).

### All output files
```
novel_books/
  <book-id>/
    <original>_with_generated.epub         ← full EPUB with new chapters appended
    master_bible.md                        ← canon reference document (edit freely)
    world_state.json                       ← live character/plot/lore state (edit freely)
    generated_chapters_combined.md         ← all new chapters in one markdown file
    generated_chapters/
      chapter_0001_plan.md                 ← AI's structural plan (useful for debugging)
      chapter_0001.md                      ← generated chapter prose
      chapter_0002_plan.md
      chapter_0002.md
      ...

chroma_books/
  <book-id>/                               ← vector embeddings (managed automatically)
```

---

## Managing Books

All book management is handled by `book_manager.py`.

### List all books
```bash
python3 book_manager.py list
```
Shows every book with its chapter count, how many chapters have been generated, disk usage, and last modified date.

### Detailed info for a specific book
```bash
python3 book_manager.py info cultivation_story
```
Shows world state details: active characters, plot threads, last chapter hook, current arc, and disk usage.

### Delete a book (with confirmation prompt)
```bash
python3 book_manager.py delete cultivation_story
```
Permanently removes everything — the `novel_books/<id>` folder and the `chroma_books/<id>` vector store. You will be asked to type `yes` to confirm before anything is deleted.

### Delete a book without being prompted
```bash
python3 book_manager.py delete cultivation_story --yes
```

---

## How It Works

Every run executes in two phases:

### Phase 1 — Analyse the existing novel (runs once per book)

| Step | What happens |
|------|-------------|
| 1. Parse | Detects chapter headers and splits the file into individual chapters |
| 2. Chapter summaries | Every chapter gets a 150–250 word dense summary capturing characters, actions, power progression, and hooks |
| 3. Arc summaries | Every 20 chapters are grouped into a 500–700 word arc-level summary |
| 4. World state | Builds a live JSON file tracking characters, active/resolved plot threads, world rules, factions, key items, and the last chapter's hook |
| 5. Master bible | Synthesizes everything into a 1000–1800 word canon reference document |
| 6. Vector index | All chapters are embedded and stored locally in ChromaDB |

### Phase 2 — Generate N chapters (loops per chapter)

| Step | What happens |
|------|-------------|
| 7. RAG retrieval | Queries ChromaDB with the current story context to surface the most relevant past scenes |
| 8. Chapter planning | A dedicated LLM call produces a full structural plan: opening hook, 3–5 scenes, thread advancement, character moment, conflict beat, revelations, ending hook, continuity traps |
| 9. Chapter writing | Writes 1800–2500 words of prose using the plan, master bible, last 3 full chapters verbatim, and RAG context |
| 10. State update | Summarises the new chapter, updates world state JSON, indexes the new chapter into ChromaDB. Refreshes master bible every 5 chapters |

---

## Tips

**The ChromaDB index persists between runs.** The first run for a large novel (e.g. 300 chapters) will take a while to summarise and index everything. Every subsequent run for that book-id reuses the existing index and only processes new chapters.

**Edit `world_state.json` manually** to fix any character or lore details the AI got wrong during extraction. This file is the live source of truth the generator reads every chapter.

**Edit `master_bible.md` manually** to inject author intent, stylistic notes, or plot directions you want the generator to follow. It is reread every generation run.

**Inspect `chapter_XXXX_plan.md`** before reading a generated chapter. The plan shows the AI's structural reasoning and is the first place to look if a chapter feels off — the fix is usually to edit the plan and regenerate.

**Increase `TARGET_WORDS`** in `novel_generator.py` (line near the top) if you want longer chapters:
```python
TARGET_WORDS = (2500, 3500)
```

---

## Cost Estimate

DeepSeek V3 on OpenRouter costs approximately **$0.27 / million input tokens** and **$1.10 / million output tokens**.

| Novel size | Initial load | Per new chapter | 10 new chapters |
|---|---|---|---|
| 50 chapters | ~$0.15 | ~$0.04 | ~$0.55 |
| 200 chapters | ~$0.60 | ~$0.06 | ~$1.20 |
| 500 chapters | ~$1.50 | ~$0.08 | ~$2.30 |

The initial load cost is a one-time cost per book. Subsequent sessions for the same book-id skip re-summarisation and cost only the per-chapter rate.

---

## Troubleshooting

**`command not found: python3`**
Install Python via Homebrew: `brew install python@3.11`

**`ModuleNotFoundError`**
Make sure your virtual environment is active: `source venv/bin/activate`, then re-run `pip install -r requirements.txt`

**Embedding model download is slow**
The BAAI/bge-large-en-v1.5 model is ~1.3 GB and only downloads once. It is cached by HuggingFace in `~/.cache/huggingface/`.

**EPUB output looks wrong in Apple Books**
Open the file in [Calibre](https://calibre-ebook.com) first and run "Check Book" to see if there are any structural issues. The ebooklib output is valid EPUB3 but some readers are stricter than others.

**API rate limit errors**
The script automatically retries with exponential backoff (up to 3 attempts). If you are hitting persistent rate limits, add a delay between chapter generations by increasing the sleep time in `LLMClient.complete()`.
