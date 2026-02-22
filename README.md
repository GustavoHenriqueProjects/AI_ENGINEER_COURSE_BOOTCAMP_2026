# AI Engineer Course Bootcamp 2026

Repositório de estudos do bootcamp de Engenharia de IA 2026, contendo lições, exemplos práticos e projetos desenvolvidos durante o curso.

## 📚 Estrutura do Projeto

```
AI_ENGINEER_COURSE_BOOTCAMP_2026/
├── SECTION21/          # Processamento de Linguagem Natural (NLP)
└── README.md          # Este arquivo
```

## 📖 Seções do Curso

- **SECTION21**: Processamento de Linguagem Natural (NLP) - Normalização de texto e análise de sentimento

> 💡 **Nota**: Cada seção possui seu próprio README com instruções detalhadas de configuração, instalação e uso. Consulte o README específico de cada seção para mais informações.

## 📦 Dependências

Cada seção possui seu próprio arquivo `requirements.txt` com as dependências específicas. Consulte o README de cada seção para instruções de instalação.

## 🤝 Contribuindo

Este é um repositório de estudos pessoal. Sinta-se à vontade para fazer fork e adaptar para seus próprios estudos.

## 📝 Licença

Este projeto é para fins educacionais.

## Estrutura do projeto
ChatBot/
├── app.py              # Só UI e fluxo do chat (~55 linhas)
├── config.py           # Constantes e system prompt
├── services/
│   ├── __init__.py     # Expõe transcrever_audio e stream_chat_generator
│   ├── transcriber.py  # Whisper (transcrição de áudio)
│   └── llm.py          # Cliente OpenAI/Ollama e stream do chat
├── requirements.txt
└── tools.txt
