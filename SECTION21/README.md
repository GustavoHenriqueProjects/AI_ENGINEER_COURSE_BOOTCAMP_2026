# SECTION 21 - Processamento de Linguagem Natural (NLP)

Esta seção aborda conceitos fundamentais de Processamento de Linguagem Natural (NLP), com foco em normalização de texto e preparação de dados para análise.

## 📋 Conteúdo da Lição

### Normalização de Texto

A normalização de texto é uma etapa crucial no pré-processamento de dados de NLP. Nesta lição, aprendemos a:

- Converter texto para minúsculas (lowercase)
- Aplicar normalização em strings individuais
- Processar listas de frases de forma eficiente

## 📁 Arquivos da Seção

- `exemplo.py` - Script Python com exemplos de normalização de texto
- `exemplo.ipynb` - Notebook Jupyter interativo com os mesmos exemplos
- `requirements.txt` - Dependências necessárias para esta seção

## 🛠️ Instalação

### 1. Criar e Ativar Ambiente Virtual

```bash
conda create --name section21 python=3.11
conda activate section21
```

### 2. Instalar Dependências

As dependências incluem bibliotecas essenciais para NLP:

```bash
pip install "numpy<2.0" nltk==3.9.1 pandas==2.2.3 matplotlib==3.10.0 spacy==3.8.3 textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 scikit-learn==1.6.0 gensim==4.3.3 seaborn==0.13.2 torch==2.5.1 ipywidgets==8.1.5 chardet ipykernel jupyterlab notebook
```

Ou instale diretamente do arquivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Instalar Modelo de Linguagem do spaCy

```bash
python -m spacy download en_core_web_sm
```

### 4. Configurar Jupyter Kernel (Opcional)

Se for usar o Jupyter Notebook:

```bash
python -m ipykernel install --user --name=section21
```

## 💻 Exemplos de Código

### Normalização de String Individual

```python
sentence = "Her cat's name is Luma"
lower_sentence = sentence.lower()
print(lower_sentence)
# Output: her cat's name is luma
```

### Normalização de Lista de Frases

```python
sentence_list = ['Could you pass me the tv remote',
                'It is IMPOSSIBLE to find this hotel', 
                'What is the weather in Tokyo']

lower_sentence_list = [x.lower() for x in sentence_list]
print(lower_sentence_list)
# Output: ['could you pass me the tv remote', 'it is impossible to find this hotel', 'what is the weather in tokyo']
```

## 🎯 Objetivos de Aprendizado

Ao final desta seção, você será capaz de:

- ✅ Entender a importância da normalização de texto em NLP
- ✅ Aplicar transformações de lowercase em strings Python
- ✅ Processar listas de texto de forma eficiente usando list comprehensions
- ✅ Preparar dados de texto para análises subsequentes

## 📚 Bibliotecas Utilizadas

- **spaCy**: Processamento avançado de linguagem natural
- **NLTK**: Natural Language Toolkit para processamento de texto
- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **transformers**: Modelos de linguagem pré-treinados
- **scikit-learn**: Machine learning e análise de dados

## 🔄 Próximos Passos

Após dominar a normalização básica de texto, os próximos tópicos podem incluir:

- Tokenização
- Remoção de stop words
- Stemming e Lemmatization
- Análise de sentimento
- Extração de entidades nomeadas

## 📝 Notas

- A normalização de texto é essencial para garantir consistência nos dados
- Converter para minúsculas ajuda a reduzir a dimensionalidade do vocabulário
- Sempre normalize os dados antes de aplicar algoritmos de NLP


