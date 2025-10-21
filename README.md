# Documentação Teórica do Projeto

## 1. TF-IDF (Term Frequency-Inverse Document Frequency)

O TF-IDF é uma técnica amplamente utilizada em recuperação de informações e mineração de texto. Ele mede a importância de uma palavra em um documento em relação a um conjunto de documentos (corpus). O cálculo é feito em duas partes:

- **Term Frequency (TF)**: A frequência de um termo em um documento. É calculada como o número de vezes que um termo aparece em um documento dividido pelo total de termos no documento.

- **Inverse Document Frequency (IDF)**: Mede a importância de um termo em todo o corpus. É calculado como o logaritmo do número total de documentos dividido pelo número de documentos que contêm o termo.

A fórmula do TF-IDF é:
\[ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) \]

onde \( t \) é o termo e \( d \) é o documento.

## 2. Similaridade por Cosseno

A similaridade por cosseno é uma medida que calcula a similaridade entre dois vetores, medindo o cosseno do ângulo entre eles. É frequentemente utilizada em sistemas de recomendação para medir a similaridade entre documentos ou entre um usuário e um item.

A fórmula é:
\[ \text{similaridade}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} \]

onde \( A \) e \( B \) são os vetores que representam os documentos ou usuários.

## 3. Estratégia Híbrida

A estratégia híbrida combina diferentes abordagens de recomendação para melhorar a precisão e a relevância das recomendações. Neste projeto, a combinação é feita entre:

- **Recomendações baseadas em conteúdo**: Utilizando TF-IDF e similaridade por cosseno para encontrar itens semelhantes com base nas características dos jogos.
  
- **Recomendações baseadas em perfil de usuário**: Considerando as avaliações e preferências dos usuários para ajustar as recomendações.

Um parâmetro ajustável \( \alpha \) é utilizado para ponderar a contribuição de cada abordagem na recomendação final:
\[ \text{recomendação final} = \alpha \times \text{recomendação por conteúdo} + (1 - \alpha) \times \text{recomendação por perfil} \]

## 4. Experimentos

Os experimentos realizados incluem a avaliação da qualidade das recomendações utilizando métricas como Precision@K e Recall@K. Os resultados foram analisados para determinar a eficácia das diferentes configurações de TF-IDF e a influência do parâmetro \( \alpha \) na estratégia híbrida.

### Resultados

Os resultados dos experimentos devem ser documentados aqui, incluindo gráficos e tabelas que mostram a performance das diferentes configurações.
