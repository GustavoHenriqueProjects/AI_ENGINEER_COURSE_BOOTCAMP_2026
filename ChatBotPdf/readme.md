# Chatbot com Ollama

Chatbot em Python com **Streamlit** (frontend) e **FastAPI** (backend): chat com Ollama e rota `/rag` com busca em **Chroma** + geração no modelo escolhido.

## Estrutura do projeto


| Caminho                                                        | Função                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------ |
| `[app.py](app.py)`                                             | Interface Streamlit                                    |
| `[api/api.py](api/api.py)`                                     | FastAPI: `/chat`, `/rag`, `/config`                    |
| `[api/config.py](api/config.py)`                               | URLs, Chroma, embeddings, textos da assistente         |
| `[api/schemas.py](api/schemas.py)`                             | Modelos Pydantic dos requests                          |
| `[api/knowledge_loader.py](api/knowledge_loader.py)`           | Lê o manual em blocos (cabeçalho `CHAVE=valor`)        |
| `[api/rag_index.py](api/rag_index.py)`                         | Indexação Chroma (`build_chroma_index`, abrir o vetor) |
| `[api/services/chat_service.py](api/services/chat_service.py)` | Streaming para Ollama / OpenAI / Gemini                |
| `[api/services/rag_service.py](api/services/rag_service.py)`   | Pipeline `/rag` (só consulta o Chroma já persistido)   |
| `[build_rag.py](build_rag.py)`                                 | **CLI que recria o índice Chroma**                     |
| `[txt/manual_brain4care.txt](txt/manual_brain4care.txt)`       | Base textual padrão para o RAG (`RAG_KNOWLEDGE_FILE`)  |


**`/rag` (resumo):** `use_rag=true` → resposta alinhada ao contexto recuperado; `use_rag=false` → contexto do manual como apoio e o modelo pode usar conhecimento geral. Ajuste no Streamlit conforme o app.

## Requisitos

- Python 3.12+
- [Ollama](https://ollama.ai) instalado
- Dependências: `[requirements.txt](requirements.txt)`

## Setup rápido

Na raiz do repositório:

```bash
source chatbot_pdf/bin/activate
 OR
.\chatbot_pdf\Scripts\Activate.ps1

pip install -r requirements.txt
```

Opcional: copie `[.env.example](.env.example)` para `.env` e ajuste variáveis (URLs, chaves opcionais, `RAG_KNOWLEDGE_FILE`, etc.).

## Recriar o índice Chroma

O endpoint `/rag` **não** reindexa a cada requisição: o banco fica em disco (por padrão a pasta `db/`). Sempre que alterar o manual ou o modelo de embedding, reconstrua o índice:

```bash
source chatbot_pdf/bin/activate
python build_rag.py
```

O script **apaga** o diretório configurado em `CHROMA_PERSIST_DIR` e grava um Chroma novo a partir de `RAG_KNOWLEDGE_FILE` (padrão: `txt/manual_brain4care.txt`). Sem esse passo, `/rag` responde que o índice está ausente.

Variáveis relevantes (ver `[api/config.py](api/config.py)`): `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL`, `RAG_KNOWLEDGE_FILE`, `RAG_METADATA_KEY_LANG` (`pt` ou `en` — nomes das chaves de metadados no Chroma; reindexar ao mudar), e para o `/rag`: **`RAG_K`**, **`RAG_CONTEXT_MAX_CHARS`**, **`RAG_TEMPERATURE`** (Ollama; recomendado **0,0–0,3** para respostas fiéis ao manual — padrão **0,2**). O body do `/rag` pode enviar `top_k` ou `temperature` para sobrescrever esses valores por requisição.

### `EMBEDDING_MODEL` (embeddings do RAG)

O padrão no código é **`BAAI/bge-m3`** (modelo [open source](https://huggingface.co/BAAI/bge-m3) da BAAI, multilíngue, forte em recuperação). Funciona bem em máquinas com RAM moderada (por exemplo 16 GB) e GPU consumer; depende de sentence-transformers / PyTorch (não usa Ollama para vetorizar).

Para **pouca RAM** ou indexação mais leve na CPU, use variáveis de ambiente para trocar o modelo:

- **`intfloat/multilingual-e5-small`** — menor footprint de memória, ainda útil para RAG multilíngue.
- **`intfloat/multilingual-e5-base`** — meio-termo entre custo de RAM e qualidade.

Nesses modelos E5 o backend aplica automaticamente os prefixos `query:` / `passage:` esperados pelo treino. Ao mudar qualquer modelo de embedding, apague `CHROMA_PERSIST_DIR` e rode `python build_rag.py` de novo.

## Rodar

**Backend** (outro terminal, com o venv ativo, na pasta `ChatBotPdf/`):

```bash
uvicorn api.api:app --reload --port 8000 --reload-dir api
```

Use `--reload-dir api` para o watcher **não** incluir o venv (`chatbot_pdf/Lib/site-packages/…`). Caso contrário, arquivos do PyTorch podem ser detectados como “alterados” e o servidor reinicia em loop (`WatchFiles detected changes in … torch …`).

**Ollama** — instale um modelo, por exemplo:

```bash
ollama pull llama3.2:1b
ollama pull bge-m3
```

**Frontend:**

```bash
streamlit run app.py
```

O app conversa com `http://localhost:8000` (ajuste no Streamlit se mudar a porta).

## Troubleshooting

- **Ollama não responde** — confira se o serviço está em execução (ex.: `http://localhost:11434`) e se o modelo já foi baixado (`ollama pull …`).
- **`/rag` sem índice** — rode `python build_rag.py` com o venv ativado (veja seção acima).
- **Trocou `EMBEDDING_MODEL`** — apague a pasta do Chroma e execute `python build_rag.py` de novo.

### Testar o Chroma no terminal

Na pasta **`ChatBotPdf/`** (onde está o pacote `api/`), com o venv ativo. `doc.page_content` é o atributo correto do LangChain.

**Linux / macOS (bash)** — heredoc:

```bash
source chatbot_pdf/bin/activate
python <<'PY'
from api.config import RAG_CONTEXT_MAX_CHARS, RAG_K
from api.rag_index import chroma_index_exists, open_chroma_vectorstore

if not chroma_index_exists():
    raise SystemExit("Sem índice: rode python build_rag.py")

db = open_chroma_vectorstore()
q = "Qual o endereço em São Paulo?"
for doc, score in db.similarity_search_with_score(q, k=RAG_K):
    print(score, doc.metadata)
    body = doc.page_content or ""
    print((body[:RAG_CONTEXT_MAX_CHARS] if RAG_CONTEXT_MAX_CHARS > 0 else body), "\n")
PY
```

**Windows (PowerShell)** — o PowerShell **não** suporta `<<'PY'`; use *here-string* enviado ao Python:

```powershell
cd ChatBotPdf   # se ainda não estiver nesta pasta
.\chatbot_pdf\Scripts\Activate.ps1
@'
from api.config import RAG_CONTEXT_MAX_CHARS, RAG_K
from api.rag_index import chroma_index_exists, open_chroma_vectorstore

if not chroma_index_exists():
    raise SystemExit("Sem índice: rode python build_rag.py")

db = open_chroma_vectorstore()
q = "Qual o endereço em São Paulo?"
for doc, score in db.similarity_search_with_score(q, k=RAG_K):
    print(score, doc.metadata)
    body = doc.page_content or ""
    print((body[:RAG_CONTEXT_MAX_CHARS] if RAG_CONTEXT_MAX_CHARS > 0 else body), "\n")
'@ | python -
```

Os valores vêm de **`RAG_K`** e **`RAG_CONTEXT_MAX_CHARS`** em `api/config.py` (ou `.env`).
