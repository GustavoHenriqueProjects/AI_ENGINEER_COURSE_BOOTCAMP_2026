"""ChatBot Brain4Care — interface Streamlit com suporte a texto e áudio."""

import base64
import os
import re
import sys
import time
from pathlib import Path

# Garante que o diretório ChatBot está no path (para imports de config e services)
_chatbot_dir = Path(__file__).resolve().parent
if str(_chatbot_dir) not in sys.path:
    sys.path.insert(0, str(_chatbot_dir))

import streamlit as st


def _stream_text(text: str, delay_per_word: float = 0.03):
    """Gera o texto palavra por palavra para efeito de digitação."""
    words = text.split()
    for i, w in enumerate(words):
        yield w + (" " if i < len(words) - 1 else "")
        time.sleep(delay_per_word)

import config as _config
MODEL = _config.MODEL
SYSTEM_PROMPT = _config.SYSTEM_PROMPT
GREETING_RESPONSE = _config.GREETING_RESPONSE
FAQ_RESPOSTAS = _config.FAQ_RESPOSTAS
FALLBACK_SUPPORT_MSG = getattr(_config, "FALLBACK_SUPPORT_MSG", "No momento não consegui processar sua solicitação. Por favor, entre em contato com o suporte técnico da Brain4care para obter a informação desejada.")
LOGO_PATH = _config.LOGO_PATH
BACKGROUND_PATH = _config.BACKGROUND_PATH
from services import transcrever_audio, stream_chat_generator, get_relevant_context
from services.rag import is_composition_question, is_person_question, is_year_question


# Palavras ignoradas na hora de casar pergunta do usuário com o FAQ
_FAQ_STOPWORDS = {
    "qual", "é", "o", "a", "os", "as", "da", "de", "do", "em", "no", "na", "para",
    "com", "que", "e", "dos", "das", "um", "uma", "onde", "quem", "quantos", "quando",
    "tem", "foi", "do",
}


def _words(text: str) -> set:
    """Extrai palavras significativas (min 2 chars, fora stopwords)."""
    t = re.sub(r"[^\w\sàáâãäéèêëíìîïóòôõöúùûüç]", " ", text.lower())
    return {w for w in t.split() if len(w) >= 2 and w not in _FAQ_STOPWORDS}


def _get_faq_answer(user_message: str) -> str | None:
    """
    Se a pergunta do usuário bater com alguma do FAQ, retorna a resposta exata.
    Assim não precisa chamar RAG nem LLM — resposta na hora.
    """
    if not user_message or not FAQ_RESPOSTAS.strip():
        return None
    user_w = _words(user_message)
    if not user_w:
        return None
    best_score = 0
    best_resposta = None
    for line in FAQ_RESPOSTAS.strip().split("\n"):
        line = line.strip()
        if " → " not in line:
            continue
        pergunta, resposta = line.split(" → ", 1)
        pergunta = pergunta.strip().lstrip("•").strip()
        resposta = resposta.strip()
        pergunta_w = _words(pergunta)
        if not pergunta_w:
            continue
        overlap = len(pergunta_w & user_w)
        # Perguntas curtas (ex: "Quem é o CEO?" → só "ceo"): basta 1 palavra em comum
        min_overlap = min(2, len(pergunta_w))
        if overlap >= min_overlap and overlap > best_score:
            best_score = overlap
            best_resposta = resposta
    return best_resposta


def _is_simple_greeting(text: str) -> bool:
    """Detecta saudações (ex: Bom dia, Olá, Bom dia meu nome é X) para responder na hora sem RAG/LLM."""
    if not text:
        return False
    t = text.strip().lower()
    t_clean = re.sub(r"[.!?]+$", "", t)
    greetings_exact = (
        "bom dia", "boa tarde", "boa noite", "olá", "ola", "oi", "hey", "e aí", "e ai",
        "oi tudo bem", "ola tudo bem", "olá tudo bem", "fala", "salve",
    )
    if t_clean in greetings_exact:
        return True
    if not t.startswith(("bom dia", "boa tarde", "boa noite", "olá ", "ola ", "oi ", "hey ", "e aí ", "e ai ")):
        return False
    # Mensagem curta: só saudação
    if len(t) <= 25:
        return True
    # Até ~60 chars e parece apresentação ("meu nome é", "sou o", "me chamo") → saudação, resposta rápida
    if len(t) <= 60 and "?" not in text:
        if any(x in t for x in ("meu nome", "me chamo", "sou o ", "sou a ", "sou eu ")):
            return True
    return False


def _extract_name_from_greeting(text: str) -> str | None:
    """Se a mensagem for uma saudação com apresentação, extrai o nome (ex: 'meu nome é Gustavo' → Gustavo)."""
    t = text.strip()
    match = re.search(
        r"(?:meu\s+nome\s+é|me\s+chamo|sou\s+o\s+|sou\s+a\s+)\s*([A-Za-zÀ-ÿ\s]+?)(?:\s*[.!?,]|$)",
        t,
        re.IGNORECASE,
    )
    if match:
        name = match.group(1).strip()
        if name and len(name) < 50:
            return name.title()
    return None


def _greeting_opening(user_message: str) -> str:
    """Retorna a saudação que corresponde à do usuário (Bom dia, Boa tarde, Boa noite ou Olá)."""
    t = user_message.strip().lower()
    if t.startswith("boa noite"):
        return "Boa noite!"
    if t.startswith("boa tarde"):
        return "Boa tarde!"
    if t.startswith("bom dia"):
        return "Bom dia!"
    return "Olá!"


def _greeting_response_with_name(user_message: str) -> str:
    """Resposta de saudação, personalizada com o nome e o período do dia."""
    opening = _greeting_opening(user_message)
    base = (
        f"{opening} Eu sou a Fernanda, assistente especialista da Brain4Care. "
        "Estou aqui para ajudá-lo com qualquer dúvida que tenha sobre a tecnologia não invasiva "
        "projetada para o cuidado e monitoramento da saúde neurológica."
    )
    name = _extract_name_from_greeting(user_message)
    if name:
        return f"{opening} {name}! Eu sou a Fernanda, assistente especialista da Brain4Care. Estou aqui para ajudá-lo com qualquer dúvida que tenha sobre a tecnologia não invasiva projetada para o cuidado e monitoramento da saúde neurológica."
    return base

st.set_page_config(
    page_title="Assistente Brain4care",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS — identidade visual Brain4care + background
_background_b64 = ""
if os.path.isfile(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, "rb") as f:
        _background_b64 = base64.b64encode(f.read()).decode("utf-8")

_background_css = ""
if _background_b64:
    _background_css = """
    [data-testid="stAppViewContainer"] {
        background-image: url("data:image/webp;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.82) 0%%, rgba(240,249,250,0.88) 100%%);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stAppViewContainer"] > section { position: relative; z-index: 1; }
    """ % _background_b64

st.markdown(
    """
    <style>
    /* Título e cabeçalho em azul navy */
    h1 { color: #0A3572 !important; }
    /* Borda e destaques em ciano/teal */
    .stChatInput input:focus, .stChatInput textarea:focus { box-shadow: 0 0 0 2px #2EC5D3 !important; }
    div[data-testid="stHorizontalBlock"] > div:first-child img { max-height: 56px; object-fit: contain; }
    %s
    </style>
    """ % _background_css,
    unsafe_allow_html=True,
)

# Logo e identidade visual Brain4care
if os.path.isfile(LOGO_PATH):
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(LOGO_PATH, width="stretch")
    with col_title:
        st.title("Assistente Brain4care")
        st.caption("Monitorização não invasiva da dinâmica intracraniana — pergunte à Fernanda.")
else:
    st.title("Assistente Brain4care")

# Estado da sessão
if "openai_model" not in st.session_state:
    st.session_state.openai_model = MODEL
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Histórico do chat
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input: texto ou áudio
prompt = st.chat_input("Escreva ou grave sua mensagem aqui", accept_audio=True)

user_message = None
if prompt:
    if isinstance(prompt, str):
        user_message = prompt.strip()
    elif getattr(prompt, "text", None):
        user_message = prompt.text.strip()
    elif getattr(prompt, "audio", None):
        with st.spinner("Transcrevendo áudio..."):
            user_message = transcrever_audio(prompt.audio.read())

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        # Saudações: resposta fixa com efeito de digitação
        if _is_simple_greeting(user_message):
            response = _greeting_response_with_name(user_message)
            response = st.write_stream(_stream_text(response))
        # Perguntas do FAQ: resposta com efeito de digitação (mantém padrão do LLM)
        elif (faq_answer := _get_faq_answer(user_message)):
            response = st.write_stream(_stream_text(faq_answer))
        else:
            with st.spinner("Pensando ..."):
                try:
                    messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    # RAG: injeta contexto dos PDFs da empresa só nesta chamada
                    context = get_relevant_context(user_message)
                    if context:
                        if is_composition_question(user_message):
                            full_context = context
                            augmented = (
                                "Liste TODOS os nomes e cargos que constam no contexto abaixo. Não omita nenhum.\n\n"
                                f"Contexto:\n\n{full_context}\n\n"
                                f"Pergunta: {user_message}"
                            )
                        elif is_person_question(user_message):
                            augmented = (
                                "Use APENAS o contexto abaixo. Responda no formato 'Nome — Cargo' com os cargos que aparecem no contexto. "
                                "Se a pessoa aparece em mais de um lugar (ex.: Diretoria e Conselho), cite TODOS os cargos. "
                                "Não use informações de outras fontes.\n\n"
                                f"Contexto:\n\n{context}\n\n"
                                f"Pergunta: {user_message}"
                            )
                        elif is_year_question(user_message):
                            augmented = (
                                "Use APENAS o contexto abaixo. Responda com a informação exata que consta no contexto sobre o ano mencionado. "
                                "Não invente nem use informações de outras fontes. Se o contexto não mencionar o ano, diga que não tem essa informação.\n\n"
                                f"Contexto:\n\n{context}\n\n"
                                f"Pergunta: {user_message}"
                            )
                        else:
                            full_context = FAQ_RESPOSTAS.strip() + "\n\n---\n\n" + context
                            augmented = (
                                "Responda com base no contexto abaixo (perguntas frequentes e material da empresa). "
                                "Se a informação estiver no contexto, use-a na resposta. "
                                "Se NÃO estiver no contexto, diga que você não tem essa informação e oriente o usuário a entrar em contato com o suporte técnico. "
                                "Não invente informações.\n\n"
                                f"Contexto da empresa:\n\n{full_context}\n\n"
                                f"Pergunta do usuário: {user_message}"
                            )
                        messages_for_llm = messages[:-1] + [{"role": "user", "content": augmented}]
                    else:
                        messages_for_llm = messages
                    stream = stream_chat_generator(st.session_state.openai_model, messages_for_llm)
                    response = st.write_stream(stream)
                except Exception:
                    response = st.write_stream(_stream_text(FALLBACK_SUPPORT_MSG))
    st.session_state.messages.append({"role": "assistant", "content": response})
