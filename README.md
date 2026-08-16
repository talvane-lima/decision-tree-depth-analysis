# Análise de Overfitting: Árvores de Decisão vs Random Forest

Este repositório contém um experimento prático desenvolvido em Python para demonstrar visualmente como o aumento da complexidade (profundidade) em uma Árvore de Decisão leva ao **overfitting** — caracterizado pela memorização extrema dos dados de treino (a chamada "decoreba") —, e como a utilização de um ensemble (Random Forest) mitiga esse problema.

## 📊 Dataset Utilizado
O experimento utiliza o dataset tabular clássico **Adult Census Income** (OpenML ID: 1590), onde o objetivo é prever se a renda de um indivíduo é maior ou menor que $50K anuais, baseado em variáveis como idade, educação, horas trabalhadas, etc.

## 📁 Estrutura de Pastas

A estrutura do projeto é a seguinte:

```text
📦 decision-tree-depth-analysis
 ┣ 📂 plots/                         # Diretório contendo todas as visualizações e gráficos gerados
 ┃ ┣ 📜 figura1_desempenho.png       # Acurácia de Treino vs Teste em Árvores de Decisão
 ┃ ┣ 📜 figura1b_desempenho_rf.png   # Acurácia de Treino vs Teste em Random Forest
 ┃ ┣ 📜 figura2_gap_generalizacao... # Diferença entre Treino/Teste por profundidade (DT)
 ┃ ┣ 📜 figura2b_gap_generalizacao...# Diferença entre Treino/Teste por profundidade (RF)
 ┃ ┣ 📜 figura3_complexidade.png     # Crescimento exponencial de Nós e Folhas (DT)
 ┃ ┣ 📜 figura4_top_features.png     # Importância das Top 10 variáveis
 ┃ ┣ 📜 figura6_evolucao_cortes...   # Histograma revelando a fragmentação ("decoreba")
 ┃ ┣ 📜 figura7_tree_depth_X.png     # Árvores de Decisão desenhadas (Profundidades 2 e 4)
 ┃ ┣ 📜 figura8_comparacao_rf.png    # Comparação de Performance Geral
 ┃ ┗ 📜 figura9_feature_importance...# Distribuição das variáveis mais importantes no Ensemble
 ┣ 📜 experiment.py                  # Script Python principal para rodar todo o pipeline e gerar as imagens
 ┣ 📜 metrics_by_depth.csv           # Tabela com as métricas geradas por cada profundidade da Árvore
 ┣ 📜 report.md                      # Relatório técnico completo detalhando as conclusões
 ┗ 📜 README.md                      # Este arquivo
```

## 🚀 Como executar o Experimento

As dependências necessárias para o projeto são bibliotecas padrão de ciência de dados no Python.

1. Instale as bibliotecas:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Execute o script principal. Ele fará o download do dataset, treinará os modelos e regerará todos os gráficos e métricas na pasta `plots/`:
   ```bash
   python experiment.py
   ```

## 🧠 Principais Resultados Encontrados

O experimento chegou às seguintes conclusões práticas, exploradas em mais detalhes no arquivo [`report.md`](report.md):

1. **O Ponto de Inflexão (Overfitting)**: Árvores rasas (profundidade 2 a 6) conseguem criar regras gerais. A partir da profundidade 8~12, a Árvore de Decisão contínua decorando os dados de treino: a acurácia de treino beira 100%, mas a de teste para de crescer, gerando um imenso **Gap de Generalização**.
2. **"Decoreba" Visual (Figura 6)**: Na profundidade máxima, vemos que a árvore escolhe a variável `age` e faz dezenas de minúsculos cortes de limite, criando uma regra específica para quase cada pessoa individual. Em vez de criar um balde para "Adultos" ou "Idosos", ela fatiou os dados isolando pessoas de "33 anos", "34 anos", demonstrando a total falta de generalização.
3. **A Força da Random Forest**: Ao testar a Random Forest nas mesmas profundidades (Figuras 1b e 2b), a curva de overfitting praticamente desaparece. O gap de generalização fica controlado e o algoritmo se prova incrivelmente mais robusto, estabilizando as features importantes e neutralizando o ruído capturado por árvores isoladas.
