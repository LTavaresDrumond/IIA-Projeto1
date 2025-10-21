# Sistema de Recomendação de Jogos Usando TF-IDF e Estratégia Híbrida
Projeto desenvolvido para a disciplina de Inteligência Artificial

## Sumário
1. [Introdução](#introdução)
2. [Metodologia](#metodologia)
3. [Implementação](#implementação)
4. [Resultados e Discussão](#resultados-e-discussão)
5. [Conclusão](#conclusão)

## Introdução

Este projeto implementa um sistema de recomendação de jogos que combina análise de conteúdo (usando TF-IDF) com filtragem colaborativa. O sistema permite que usuários:
- Avaliem jogos em uma escala de 0-5
- Busquem jogos por descrição textual
- Recebam recomendações personalizadas baseadas em seu perfil
- Explorem recomendações híbridas que combinam diferentes estratégias

### Objetivos
- Desenvolver um sistema de recomendação eficiente e explicável
- Comparar diferentes abordagens de pré-processamento textual
- Avaliar o impacto de diferentes parâmetros nas recomendações
- Implementar uma interface interativa para testes e validação

## Metodologia

### 1. Preparação dos Dados
- Fonte: Dataset de jogos com características textuais
- Pré-processamento:
  - Normalização de texto
  - Remoção de stopwords (português e inglês)
  - Implementação de n-grams (unigrams e bigrams)
  - Matriz de utilidade para avaliações dos usuários

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
Implementamos TF-IDF para vetorizar as características dos jogos:
- Term Frequency (TF): frequência do termo no documento
- Inverse Document Frequency (IDF): importância global do termo
- Parâmetros ajustáveis:
  - ngram_range: (1,1) ou (1,2)
  - min_df: 1-5
  - weight_map: pesos diferentes para características específicas

### 3. Similaridade por Cosseno
Utilizada para calcular similaridade entre:
- Consultas textuais e documentos
- Perfis de usuários e jogos
- Jogos entre si

### 4. Estratégia Híbrida
Combinamos múltiplas fontes de informação:
- Similaridade textual (TF-IDF)
- Histórico de avaliações
- Peso ajustável (α) entre as abordagens

## Implementação

### Componentes Principais
1. **Interface Web (Streamlit)**
   - Login de usuários
   - Avaliação de jogos
   - Busca textual
   - Visualização de recomendações

2. **Processamento de Texto**
   ```python
   def build_tfidf_for_games(df_jogos, ngram_range=(1,2), min_df=1, weight_map=None):
       docs = build_weighted_documents(df_jogos, weight_map)
       vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
       X = vec.fit_transform(docs)
       return vec, X
   ```

3. **Sistema de Recomendação**
   ```python
   def recommend_by_text(query, df_jogos, vec, X, top_n=5):
       qv = vec.transform([query])
       sims = cosine_similarity(qv, X).flatten()
       idx = sims.argsort()[::-1][:top_n]
       return [(df_jogos.loc[i, 'nome_jogo'], float(sims[i])) for i in idx]
   ```

### Avaliação Offline
Implementamos métricas para avaliar a qualidade das recomendações:
- Precision@K
- Recall@K
- MAP (Mean Average Precision)

## Resultados e Discussão

### Configurações Testadas
1. **TF-IDF Básico**
   - ngram_range=(1,1)
   - min_df=1
   - Sem pesos especiais

2. **TF-IDF Otimizado**
   - ngram_range=(1,2)
   - min_df=2
   - weight_map={'caracteristica_1':2, 'caracteristica_3':2}

3. **Híbrido com α=0.6**
   - Combina TF-IDF com perfil do usuário

### Métricas de Avaliação
[Adicione aqui uma tabela ou gráfico com os resultados das diferentes configurações]

## Conclusão

- A estratégia híbrida mostrou-se mais efetiva que abordagens individuais
- O uso de bigrams melhorou a captura de conceitos compostos
- A ponderação de características específicas (ex: gênero) impactou positivamente as recomendações

### Trabalhos Futuros
- Implementar persistência em banco de dados
- Explorar técnicas de deep learning para processamento de texto
- Adicionar análise de sentimento das avaliações

## Referências

1. [TF-IDF - Scikit Learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
2. [Streamlit Documentation](https://docs.streamlit.io/)
3. [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-0-387-85820-3)

## Apêndice: Instruções de Execução

### Requisitos
```text
streamlit
pandas
numpy
scikit-learn
```

### Como Executar
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute: `streamlit run app.py`

### Deploy
O projeto está disponível online através do Streamlit Cloud em [adicionar-link-após-deploy]
