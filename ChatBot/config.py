"""Configurações centralizadas do ChatBot."""

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
# Modelo leve e rápido (1b). 3b é mais capaz mas mais lento. Outros: "llama3.2:3b", "llama3.1:8b"
MODEL = "llama3.2:3b"

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

APRESENTAÇÃO — Use exatamente este texto SOMENTE quando o usuário perguntar quem você é ou o que faz:
"Bom dia! Eu sou a Fernanda, assistente especialista da Brain4Care. Estou aqui para ajudá-lo com qualquer dúvida que tenha sobre a tecnologia não invasiva projetada para o cuidado e monitoramento da saúde neurológica."
NÃO inclua essa apresentação em outras respostas (ex.: quando perguntarem sobre diretoria, conselho, endereço etc.). Vá direto à resposta.

PERGUNTAS FREQUENTES — Quando o usuário perguntar algo semelhante às questões abaixo, responda com a informação correspondente (use o texto após a seta):
""" + FAQ_RESPOSTAS + """

DIRETRIZES:
1. Tom: profissional, científico, acessível e empático.
2. Para as perguntas acima, use sempre as respostas indicadas; para outras dúvidas, use o contexto da empresa fornecido na conversa.
3. Se a informação não constar no contexto, diga que você não tem essa resposta e oriente o usuário a entrar em contato com o suporte técnico.
4. Em perguntas sobre composição (ex.: Diretoria Executiva, Conselho de Administração), liste TODOS os nomes e cargos que constam no contexto.
5. Em perguntas sobre anos ou datas (ex.: "o que aconteceu em 2009?"), use a informação exata do trecho que menciona esse ano. Responda diretamente com o que consta no contexto — não diga que não encontrou se a informação estiver presente.
6. Para "Quem é [nome]?", use APENAS os cargos que aparecem no contexto dos trechos (formato "Nome — Cargo"). Se a pessoa aparece em mais de um lugar (ex.: Diretoria e Conselho), cite TODOS os cargos. Ignore o FAQ para essa pergunta — use só o material dos trechos."""

# RAG — dados da empresa (PDFs) e índice vetorial
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_BASE_DIR, "assets")  # Logo e imagens do chat
LOGO_PATH = os.path.join(ASSETS_DIR, "brain4care_logo.png")  # Logo Brain4care
BACKGROUND_PATH = os.path.join(ASSETS_DIR, "Background.webp")  # Imagem de fundo do chat
DATA_DIR = os.path.join(_BASE_DIR, "data", "pdf")  # Coloque aqui os PDFs da empresa
CHROMA_DIR = os.path.join(_BASE_DIR, "chroma_db")  # Índice vetorial (persistido)
EMBEDDING_MODEL = "nomic-embed-text"  # Rode: ollama run nomic-embed-text
RAG_TOP_K = 25  # Mais trechos no contexto = mais chance de o trecho certo do PDF aparecer
RAG_CANDIDATES = 60  # Candidatos na busca; depois deduplicamos e limitamos a TOP_K
# Chunking: tamanho menor separa Diretoria Executiva do Conselho de Administração em trechos distintos
RAG_CHUNK_SIZE = 450
RAG_CHUNK_OVERLAP = 80
# Metadados aplicados a todos os documentos carregados dos PDFs (e herdados pelos chunks)
RAG_DOCUMENT_METADATA = {
    "empresa": "Brain4care",
    "site": "https://brain4.care",
    "tipo": "institucional",
}
