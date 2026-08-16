# Relatório: Overfitting em Árvores de Decisão e o Poder da Random Forest

Este relatório interpreta os resultados do experimento de classificação utilizando o dataset **Adult Census Income**, evidenciando o comportamento das Árvores de Decisão à medida que sua profundidade aumenta, além de comparar com uma Random Forest.

## 1. O Ponto de Overfitting (Desempenho e Generalização)

Observando as **Figuras 1 e 2**, notamos claramente um ponto de inflexão onde a árvore começa a overfitar.
- **Árvores Rasas (Depth 2 a 6)**: A acurácia no treino e no teste crescem juntas. O modelo está aprendendo regras gerais úteis.
- **A partir da Profundidade 8 a 12**: A acurácia de treino continua a subir (rumo a quase 100% na profundidade máxima), mas a acurácia de teste estagna e até começa a cair levemente.
- O **Gap de Generalização (Figura 2)** explode nas profundidades maiores, evidenciando que o modelo memorizou os dados de treino ao invés de extrair padrões que se aplicam a dados novos. A Figura 3 confirma isso, mostrando um crescimento exponencial no número de nós e folhas.

## 2. A Evolução da "Decoreba" (Cortes e Fragmentação)

A **Figura 4** (Feature Importances) mostra que árvores mais profundas tendem a distribuir a importância entre um número maior de variáveis, algumas delas possivelmente sendo ruído irrelevante sendo aproveitado para "decorar" amostras específicas.

A **Figura 6** evidencia visualmente o processo de memorização ao observar os cortes (thresholds) realizados sobre a principal variável contínua (como `age` ou `capital-gain`).
- Em uma árvore rasa (profundidade 2 ou 4), há poucos limiares (linhas vermelhas tracejadas), indicando que a árvore divide a variável em "baldes" amplos (ex: "Jovens", "Adultos", "Idosos"). Isso representa generalização.
- Na **árvore de profundidade máxima**, a variável é fatiada em centenas de fragmentos minúsculos. A árvore literalmente cria uma regra específica para quase cada valor numérico individual que apareceu no conjunto de treino. Isso **caracteriza memorização**: o modelo está separando instâncias individuais em vez de populações, perdendo totalmente a utilidade para previsões no mundo real.

## 3. Comparação com Random Forest

A **Random Forest** mitiga o overfitting combinando centenas de árvores de decisão.
Ao observar a **Figura 8**, vemos que a Random Forest não apenas supera as árvores individuais rasas e profundas em métricas de teste (Acurácia, AUC e F1), como também apresenta um **Gap de Generalização muito mais controlado** em relação à árvore profunda.

Isso ocorre pelo princípio de *bagging* (Bootstrap Aggregating) e pela seleção aleatória de features:
- Cada árvore da floresta "overfita" ligeiramente uma amostra diferente dos dados.
- Ao tirarmos a média (ou moda) de todas as árvores, os erros idiossincráticos (a "decoreba" de cada árvore) são cancelados, restando apenas o sinal verdadeiro e generalizável do modelo.
- A **Figura 9** demonstra que a Random Forest possui uma importância de features muito mais estável e distribuída suavemente. A Random Forest penaliza o uso exclusivo de uma única feature de "decoreba", tornando o modelo mais robusto frente a ruídos.

### Conclusão

Árvores de decisão sem poda são modelos de alta variância, propensos a fatiar o espaço de variáveis até englobar amostras individuais de treino (memorização). Controlar a profundidade (regularização) é fundamental. Quando o objetivo é performance pura e estabilidade, **modelos baseados em Ensembles, como a Random Forest**, são escolhas imensamente superiores, pois aproveitam a capacidade não-linear das árvores enquanto neutralizam a sua principal fraqueza (o overfitting).
