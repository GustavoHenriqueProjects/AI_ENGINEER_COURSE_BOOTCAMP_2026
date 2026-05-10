import html

import httpx
import streamlit as st

# Configuração da página Streamlit
st.set_page_config(page_title="Streamlit Chat", page_icon=":smiley:")
st.title("💬 Chatbot (Ollama ou API na nuvem)")

BACKEND_BASE = "http://localhost:8000"
BACKEND_RAG_URL = f"{BACKEND_BASE}/rag"
BACKEND_CONFIG_URL = f"{BACKEND_BASE}/config"


@st.cache_data(ttl=30)
def _fetch_backend_config() -> dict:
    try:
        r = httpx.get(BACKEND_CONFIG_URL, timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {
            "gemini_available": False,
            "gemini_model": "gemini-2.0-flash",
            "gemini_choice_value": "__gemini_google__",
            "openai_available": False,
        }


_cfg = _fetch_backend_config()
_gemini_val = _cfg.get("gemini_choice_value") or "__gemini_google__"
_gemini_model = _cfg.get("gemini_model") or "gemini-2.0-flash"

# Sidebar com seleção de modelo e informações
with st.sidebar:
    st.header("⚙️ Configurações")

    _ollama_models = [
        "llama3.2:1b",
        "llama3.2:3b",
        "llama3.1:8b",
        "deepseek-r1",
    ]
    _openai_models = []
    if _cfg.get("openai_available"):
        _openai_models = ["gpt-4o-mini", "gpt-4o"]

    _models = list(_ollama_models)
    if _cfg.get("gemini_available"):
        _models.append(_gemini_val)
    _models.extend(_openai_models)

    def _model_label(m: str) -> str:
        if m == _gemini_val:
            return f"Gemini — Google (automático: {_gemini_model})"
        if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
            return f"{m} (OpenAI — OPENAI_API_KEY)"
        return f"{m} (Ollama local)"

    model_name = st.selectbox("Modelo", _models, index=0, format_func=_model_label)
    use_rag = st.checkbox(
        "Modo empresa (RAG restrito)",
        value=True,
        help="Ligado: só responde com base nos documentos indexados (ou mensagem fixa). "
        "Desligado: pode usar conhecimento geral no Ollama e ainda aproveitar trechos do banco quando existirem.",
    )
    st.divider()
    st.markdown(
        "**Nuvem:** com `GOOGLE_API_KEY` ou `GEMINI_API_KEY`, aparece uma única opção Gemini "
        f"(modelo fixo no servidor: `{_gemini_model}`). "
        "`OPENAI_API_KEY` habilita modelos GPT na lista."
    )

# Inicializa histórico de mensagens na sessão, incluindo a instrução do sistema
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": ""}
    ]

# Exibe histórico de mensagens na interface de chat
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Captura o prompt do usuário e envia para o backend
if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f'<div style="white-space: pre-wrap;">{html.escape(prompt)}</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        try:
            response_text = ""

            if model_name == _gemini_val:
                _prov = "Gemini (Google)"
            elif model_name.startswith("gpt-") or model_name.startswith(("o1", "o3")):
                _prov = "OpenAI"
            else:
                _prov = "Ollama"
            with st.spinner(f"Enviando para backend ({_prov})..."):
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    if m["role"] in ("user", "assistant")
                ]
                payload = {
                    "query": prompt,
                    "model": model_name,
                    "use_rag": use_rag,
                    "messages": chat_history,
                }

                with httpx.stream("POST", BACKEND_RAG_URL, json=payload, timeout=120.0) as response:
                    response.raise_for_status()

                    response_placeholder = st.empty()
                    # O /rag devolve texto em streaming (sem newline por chunk): iter_lines() junta mal;
                    # st.markdown() trata _ e * como Markdown e estraga endereços / metadados.
                    for chunk in response.iter_text():
                        if chunk:
                            response_text += chunk
                            response_placeholder.markdown(
                                f'<div style="white-space: pre-wrap;">{html.escape(response_text)}</div>',
                                unsafe_allow_html=True,
                            )

        except httpx.ConnectError:
            st.error("❌ Erro: backend FastAPI não está rodando em http://localhost:8000")
            st.info(
                "Inicie o backend: `uvicorn api.api:app --reload --port 8000 --reload-dir api`"
            )
        except httpx.ReadTimeout:
            st.error("❌ Timeout: A resposta levou muito tempo. Tente novamente.")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
        else:
            # Salva a resposta no histórico de mensagens para manter o contexto
            st.session_state.messages.append({"role": "assistant", "content": response_text})
