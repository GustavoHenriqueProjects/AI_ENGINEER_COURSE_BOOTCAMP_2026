"""Configurações centralizadas do ChatBot."""

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
# Modelo leve e rápido. Outros: "llama3.2:3b", "llama3.1:8b", "deepseek-r1:1.5b"
MODEL = "llama3.2:1b"

# Perguntas frequentes: use estas respostas quando o usuário perguntar algo semelhante (fonte: visão institucional)
FAQ_RESPOSTAS = """
• Qual é o site da Brain4care? → O site oficial é https://brain4.care

• Onde fica a Brain4care? → A Brain4care tem presença em três escritórios: São Carlos (Brasil) — Av. Bruno Ruggiero Filho, 971 — Parque Santa Felícia Jardim, São Carlos — SP; São Paulo (Brasil) — Av. Queiroz Filho, 1560 — Vila Hamburguesa, São Paulo — SP; Atlanta (Estados Unidos) — 6470 E Johns Crossing, Suite 160, Office 114, Johns Creek, GA 30097.

• Qual o telefone da Brain4care em São Paulo? → O telefone do escritório em São Paulo é +55 (11) 4324-5305.

• Qual o telefone da Brain4care em São Carlos? → O telefone do escritório em São Carlos é +55 (16) 3501-4020.

• Quantos hospitais utilizam a tecnologia? → A tecnologia está em uso clínico em mais de 103 hospitais e clínicas e em uso em pesquisa em mais de 20 instituições.

• Quem é o CEO? → O CEO da Brain4care é Plínio Targa.

• Quando a empresa foi fundada? → A Brain4care foi fundada em 2014.

• A Brain4care tem FDA? → Sim. A Brain4care recebeu a certificação pelo FDA (Food and Drug Administration) em 2021.
"""

# Resposta fixa para saudações simples — evita chamar RAG e LLM, resposta instantânea
GREETING_RESPONSE = (
    "Bom dia! Eu sou a Fernanda, assistente especialista da Brain4Care. "
    "Estou aqui para ajudá-lo com qualquer dúvida que tenha sobre a tecnologia não invasiva "
    "projetada para o cuidado e monitoramento da saúde neurológica."
)

SYSTEM_PROMPT = """Você é a Fernanda, Assistente Especialista da Brain4Care.

APRESENTAÇÃO — Use exatamente este texto quando for a primeira mensagem do usuário ou quando perguntarem quem você é ou o que faz:
"Bom dia! Eu sou a Fernanda, assistente especialista da Brain4Care. Estou aqui para ajudá-lo com qualquer dúvida que tenha sobre a tecnologia não invasiva projetada para o cuidado e monitoramento da saúde neurológica."

PERGUNTAS FREQUENTES — Quando o usuário perguntar algo semelhante às questões abaixo, responda com a informação correspondente (use o texto após a seta):
""" + FAQ_RESPOSTAS + """

DIRETRIZES:
1. Tom: profissional, científico, acessível e empático.
2. Para as perguntas acima, use sempre as respostas indicadas; para outras dúvidas, use o contexto da empresa fornecido na conversa.
3. Se a informação não constar no contexto, diga que você não tem essa resposta e oriente o usuário a entrar em contato com o suporte técnico."""

# RAG — dados da empresa (PDFs) e índice vetorial
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_BASE_DIR, "assets")  # Logo e imagens do chat
LOGO_PATH = os.path.join(ASSETS_DIR, "brain4care_logo.png")  # Logo Brain4care
BACKGROUND_PATH = os.path.join(ASSETS_DIR, "Background.webp")  # Imagem de fundo do chat
DATA_DIR = os.path.join(_BASE_DIR, "data", "pdf")  # Coloque aqui os PDFs da empresa
CHROMA_DIR = os.path.join(_BASE_DIR, "chroma_db")  # Índice vetorial (persistido)
EMBEDDING_MODEL = "nomic-embed-text"  # Rode: ollama run nomic-embed-text
RAG_TOP_K = 20  # Mais trechos no contexto = mais chance de o trecho certo do PDF aparecer
RAG_CANDIDATES = 50  # Candidatos na busca; depois deduplicamos e limitamos a TOP_K
# Chunking: tamanho maior evita cortar listas (ex.: diretoria executiva) no meio
RAG_CHUNK_SIZE = 900
RAG_CHUNK_OVERLAP = 200
# Metadados aplicados a todos os documentos carregados dos PDFs (e herdados pelos chunks)
RAG_DOCUMENT_METADATA = {
    "empresa": "Brain4care",
    "site": "https://brain4.care",
    "tipo": "institucional",
}
