# 🧠 AgenticRAG —  RAG Assistant 

AgenticRAG is a **Retrieval-Augmented Generation (RAG)** web app built with **Django** and **LangChain**.  
It lets users upload documents (PDF or text), embed them locally using OpenAI’s embedding models, store them in a persistent **ChromaDB**, and query them via a chat-like interface powered by **GPT-4o-mini**.

---

## 🚀 Features

- 📄 Upload and embed **PDF** or **text** files
- 💾 Persistent **ChromaDB** vector store
- 🔍 Context-aware document retrieval (RAG)
- 💬 Conversational QA powered by **ChatOpenAI**
- ⚙️ Modular design (`loader`, `embeddings`, `vectorstore`, `rag_engine`)
- 🔑 Secure `.env`-based configuration
- 🧩 Fully compatible with **LangChain ≥ 0.2** and modern packages (`langchain-core`, `langchain-community`, `langchain-openai`, `langchain-chroma`)

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| Backend | Django 5.x |
| LLM / Embeddings | OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) |
| Retrieval | LangChain + ChromaDB |
| Database | Local persistent Chroma vector DB |
| Language | Python 3.12 |
| Environment | Virtualenv (`rag_env`) |

---

## 📦 Installation

### 1. Clone and enter the repo
```bash
git clone https://github.com/<yourusername>/agenticrag.git
cd agenticrag
```
### 2. Create and activate a virtual environment
```bash
python3 -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt

```

### ⚙️ Environment Setup

Create a .env file in the same folder as manage.py:
```bash
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
```

Make sure you’ve added this in manage.py so Django loads it automatically:
```bash
from dotenv import load_dotenv
load_dotenv()
```


### 🏗️ Project Structure
```bash
agenticrag/
├── manage.py
├── .env
├── ragcore/
│   ├── views.py
│   ├── urls.py
│   └── rag/
│       ├── loader.py
│       ├── embeddings.py
│       ├── vectorstore.py
│       └── rag_engine.py
└── chroma_db/  ← persisted vector database
```
### Key modules
```bash
File	Purpose
loader.py	Loads .pdf or .txt into LangChain Documents
embeddings.py	Wraps OpenAIEmbeddings with key from .env
vectorstore.py	Creates persistent Chroma vector DB
rag_engine.py	Runs the retrieval + LLM chain (RAG)
views.py	Django routes for /upload/ and /ask/ endpoints
```

### ▶️ Running the Server
```bash
python manage.py runserver
```
Visit http://localhost:8000￼
You can:
	•	Upload PDFs or text files at /upload/
	•	Ask questions at /ask/
