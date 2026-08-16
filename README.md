# Analise de Overfitting: Arvores de Decisao vs Random Forest

Este repositorio contem um experimento pratico desenvolvido em Python para demonstrar visualmente como o aumento da complexidade (profundidade) em uma Arvore de Decisao leva ao overfitting, caracterizado pela memorizacao extrema dos dados de treino (a chamada "decoreba"), e como a utilizacao de um ensemble (Random Forest) mitiga esse problema.

## Dataset Utilizado
O experimento utiliza o dataset tabular classico Adult Census Income (OpenML ID: 1590), onde o objetivo e prever se a renda de um individuo e maior ou menor que $50K anuais, baseado em variaveis como idade, educacao, horas trabalhadas, etc.

## Estrutura de Pastas

A estrutura do projeto e a seguinte:

decision-tree-depth-analysis
 - plots/
   - figura1_desempenho.png: Acuracia de Treino vs Teste em Arvores de Decisao
   - figura1b_desempenho_rf.png: Acuracia de Treino vs Teste em Random Forest
   - figura2_gap_generalizacao.png: Diferenca entre Treino/Teste por profundidade (DT)
   - figura2b_gap_generalizacao_rf.png: Diferenca entre Treino/Teste por profundidade (RF)
   - figura3_complexidade.png: Crescimento exponencial de Nos e Folhas (DT)
   - figura4_top_features.png: Importancia das Top 10 variaveis
   - figura6_evolucao_cortes_painel.png: Histograma revelando a fragmentacao ("decoreba")
   - figura7_tree_depth_X.png: Arvores de Decisao desenhadas (Profundidades 2 e 4)
   - figura8_comparacao_rf.png: Comparacao de Performance Geral
   - figura9_feature_importance_rf.png: Distribuicao das variaveis mais importantes no Ensemble
 - experiment.py: Script Python principal para rodar todo o pipeline e gerar as imagens
 - metrics_by_depth.csv: Tabela com as metricas geradas por cada profundidade da Arvore
 - report.md: Relatorio tecnico completo detalhando as conclusoes
 - README.md: Este arquivo

## Como executar o Experimento

As dependencias necessarias para o projeto sao bibliotecas padrao de ciencia de dados no Python.

1. Instale as bibliotecas:
   pip install pandas numpy scikit-learn matplotlib seaborn

2. Execute o script principal. Ele fara o download do dataset, treinara os modelos e regerara todos os graficos e metricas na pasta plots/:
   python experiment.py

## Principais Resultados Encontrados

O experimento chegou as seguintes conclusoes praticas, exploradas em mais detalhes no arquivo report.md:

1. O Ponto de Inflexao (Overfitting): Arvores rasas (profundidade 2 a 6) conseguem criar regras gerais. A partir da profundidade 8 a 12, a Arvore de Decisao continua decorando os dados de treino. A acuracia de treino beira 100%, mas a de teste para de crescer, gerando um imenso Gap de Generalizacao.
2. "Decoreba" Visual (Figura 6): Na profundidade maxima, vemos que a arvore escolhe a variavel age e faz dezenas de minusculos cortes de limite, criando uma regra especifica para quase cada pessoa individual. Em vez de criar um balde para "Adultos" ou "Idosos", ela fatiou os dados isolando pessoas de "33 anos", "34 anos", demonstrando a total falta de generalizacao.
3. A Forca da Random Forest: Ao testar a Random Forest nas mesmas profundidades (Figuras 1b e 2b), a curva de overfitting praticamente desaparece. O gap de generalizacao fica controlado e o algoritmo se prova incrivelmente mais robusto, estabilizando as features importantes e neutralizando o ruido capturado por arvores isoladas.
