import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configuração de estilo visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def create_dirs():
    os.makedirs('plots', exist_ok=True)

def load_and_preprocess_data():
    print("Carregando o dataset Adult Census Income...")
    # Adult dataset
    X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    
    # Tratando a variável alvo
    y = (y == '>50K').astype(int)
    
    # Separação em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Identificando colunas
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Colunas numéricas: {num_cols}")
    print(f"Colunas categóricas: {cat_cols}")
    
    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # Transformando os dados explicitamente para reter o nome das colunas depois
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = num_cols + cat_cols
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    return X_train_df, X_test_df, y_train.values, y_test.values, feature_names

def train_decision_trees(X_train, y_train, X_test, y_test):
    depths = [2, 4, 6, 8, 12, 20, None]
    results = []
    models = {}
    
    print("Treinando Decision Trees...")
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        y_pred_train = dt.predict(X_train)
        y_pred_test = dt.predict(X_test)
        
        y_prob_train = dt.predict_proba(X_train)[:, 1]
        y_prob_test = dt.predict_proba(X_test)[:, 1]
        
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        auc_test = roc_auc_score(y_test, y_prob_test)
        
        gen_gap = acc_train - acc_test
        nodes = dt.tree_.node_count
        leaves = dt.tree_.n_leaves
        actual_depth = dt.tree_.max_depth
        
        results.append({
            'max_depth_param': str(d) if d is not None else 'Máx',
            'actual_depth': actual_depth,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
            'gen_gap': gen_gap,
            'nodes': nodes,
            'leaves': leaves
        })
        
        models[d] = dt
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('metrics_by_depth.csv', index=False)
    print("Métricas salvas em metrics_by_depth.csv")
    return df_results, models

def plot_performance_and_gap(df_results):
    print("Gerando gráficos de desempenho...")
    
    # Transform x-axis to be categorical or handle 'None' well
    x_labels = df_results['max_depth_param'].tolist()
    x_ticks = range(len(x_labels))
    
    # Figura 1: Desempenho
    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, df_results['acc_train'], marker='o', label='Acurácia Treino', linewidth=2)
    plt.plot(x_ticks, df_results['acc_test'], marker='s', label='Acurácia Teste', linewidth=2)
    plt.xticks(x_ticks, x_labels)
    plt.xlabel('Max Depth')
    plt.ylabel('Acurácia')
    plt.title('Desempenho (Treino vs Teste) por Profundidade')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/figura1_desempenho.png', bbox_inches='tight')
    plt.close()
    
    # Figura 2: Gap de Generalização
    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, df_results['gen_gap'], marker='o', color='firebrick', linewidth=2)
    plt.xticks(x_ticks, x_labels)
    plt.xlabel('Max Depth')
    plt.ylabel('Gap de Generalização (Acc Treino - Acc Teste)')
    plt.title('Gap de Generalização por Profundidade')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/figura2_gap_generalizacao.png', bbox_inches='tight')
    plt.close()
    
    # Figura 3: Complexidade estrutural
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(x_ticks, df_results['nodes'], marker='o', color='teal', label='Número de Nós', linewidth=2)
    ax2.plot(x_ticks, df_results['leaves'], marker='s', color='darkorange', label='Número de Folhas', linewidth=2)
    
    ax1.set_xlabel('Max Depth')
    ax1.set_ylabel('Número de Nós', color='teal')
    ax2.set_ylabel('Número de Folhas', color='darkorange')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    
    plt.title('Complexidade Estrutural (Nós e Folhas) por Profundidade')
    fig.tight_layout()
    plt.savefig('plots/figura3_complexidade.png', bbox_inches='tight')
    plt.close()

def plot_feature_importances(models, feature_names):
    print("Gerando gráficos de top features por profundidade...")
    depths = list(models.keys())
    
    n_cols = 2
    n_rows = int(np.ceil(len(depths) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, d in enumerate(depths):
        model = models[d]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        features_list = [feature_names[j] for j in indices]
        sns.barplot(x=importances[indices], y=features_list, ax=axes[i], hue=features_list, palette='viridis', legend=False)
        axes[i].set_title(f'Top 10 Features - Max Depth: {d if d is not None else "Máx"}')
        axes[i].set_xlabel('Importância')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('plots/figura4_top_features.png', bbox_inches='tight')
    plt.close()

def plot_variable_fragmentation(models, X_train_df, feature_names):
    print("Gerando gráficos de fragmentação da principal variável contínua...")
    # Na maioria dos casos do Adult Census, 'age' é uma das mais importantes contínuas
    if 'age' in feature_names:
        main_var = 'age'
    else:
        # Pega a mais importante da árvore mais profunda
        model = models[None]
        main_var = feature_names[np.argmax(model.feature_importances_)]
        
    main_var_idx = feature_names.index(main_var)
    
    depths = list(models.keys())
    
    # Extrair os thresholds usados por cada árvore para a variável principal
    thresholds_by_depth = {}
    for d in depths:
        tree = models[d].tree_
        features = tree.feature
        thresholds = tree.threshold
        
        # Onde a árvore splitou usando a variável principal?
        splits_mask = features == main_var_idx
        var_thresholds = thresholds[splits_mask]
        thresholds_by_depth[d] = var_thresholds
        
    # Figura 5: Plot individual por profundidade para focar na fragmentação
    # Figura 6: Painel conjunto
    
    fig, axes = plt.subplots(len(depths), 1, figsize=(10, 3 * len(depths)))
    
    for i, d in enumerate(depths):
        ax = axes[i]
        sns.histplot(X_train_df[main_var], bins=50, color='lightsteelblue', ax=ax, edgecolor=None)
        
        # Desenhar os thresholds
        var_thresholds = thresholds_by_depth[d]
        for th in var_thresholds:
            ax.axvline(th, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            
        ax.set_title(f'Cortes em "{main_var}" (Profundidade {d if d is not None else "Máxima"}: {len(var_thresholds)} cortes)')
        ax.set_ylabel('Frequência')
        if i == len(depths) - 1:
            ax.set_xlabel(f'{main_var}')
        else:
            ax.set_xlabel('')
            
    plt.tight_layout()
    plt.savefig('plots/figura6_evolucao_cortes_painel.png', bbox_inches='tight')
    plt.close()

def render_trees(models, feature_names):
    print("Renderizando árvores...")
    depths_to_plot = [2, 4] # 6 fica enorme e difícil de ver
    
    for d in depths_to_plot:
        plt.figure(figsize=(20, 10))
        plot_tree(models[d], feature_names=feature_names, filled=True, rounded=True, 
                  class_names=['<=50K', '>50K'], proportion=True, max_depth=d, fontsize=8)
        plt.title(f'Decision Tree (Max Depth = {d})')
        plt.savefig(f'plots/figura7_tree_depth_{d}.png', bbox_inches='tight')
        plt.close()

def run_random_forest(X_train, y_train, X_test, y_test, dt_results, feature_names, models):
    print("Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_prob_test)
    gen_gap = acc_train - acc_test
    
    rf_metrics = {
        'Model': 'Random Forest',
        'Acc Test': acc_test,
        'F1 Test': f1_test,
        'AUC Test': auc_test,
        'Gen Gap': gen_gap
    }
    
    # Extrair melhor árvore
    best_dt_idx = dt_results['acc_test'].idxmax()
    best_dt = dt_results.iloc[best_dt_idx]
    
    deepest_dt = dt_results.iloc[-1]
    shallow_dt = dt_results.iloc[0]
    
    comparison_data = pd.DataFrame([
        {'Model': f'Árvore Rasa (d={shallow_dt["max_depth_param"]})', 'Acc Test': shallow_dt['acc_test'], 'F1 Test': shallow_dt['f1_test'], 'AUC Test': shallow_dt['auc_test'], 'Gen Gap': shallow_dt['gen_gap']},
        {'Model': f'Árvore Profunda (d={deepest_dt["max_depth_param"]})', 'Acc Test': deepest_dt['acc_test'], 'F1 Test': deepest_dt['f1_test'], 'AUC Test': deepest_dt['auc_test'], 'Gen Gap': deepest_dt['gen_gap']},
        {'Model': f'Melhor Árvore (d={best_dt["max_depth_param"]})', 'Acc Test': best_dt['acc_test'], 'F1 Test': best_dt['f1_test'], 'AUC Test': best_dt['auc_test'], 'Gen Gap': best_dt['gen_gap']},
        rf_metrics
    ])
    
    # Figura 8: Comparação de desempenho
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ['Acc Test', 'F1 Test', 'AUC Test', 'Gen Gap']
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]
        sns.barplot(x='Model', y=metric, data=comparison_data, ax=ax, palette='Set2')
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=15)
        
    plt.tight_layout()
    plt.savefig('plots/figura8_comparacao_rf.png', bbox_inches='tight')
    plt.close()
    
    # Figura 9: Feature Importance RF vs Deepest Tree
    print("Gerando figura 9...")
    rf_importances = rf.feature_importances_
    dt_importances = models[None].feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_importances,
        'Decision Tree (Max)': dt_importances
    }).melt(id_vars=['Feature'], var_name='Model', value_name='Importance')
    
    # Pegar as top 10 do RF para focar
    top_features = pd.DataFrame({'Feature': feature_names, 'Imp': rf_importances}).sort_values('Imp', ascending=False).head(10)['Feature']
    df_imp_top = df_imp[df_imp['Feature'].isin(top_features)]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_imp_top, x='Importance', y='Feature', hue='Model', palette='pastel')
    plt.title('Comparação de Feature Importances (Top 10 da RF)')
    plt.savefig('plots/figura9_feature_importance_rf.png', bbox_inches='tight')
    plt.close()

def main():
    create_dirs()
    X_train_df, X_test_df, y_train, y_test, feature_names = load_and_preprocess_data()
    
    dt_results, models = train_decision_trees(X_train_df, y_train, X_test_df, y_test)
    
    plot_performance_and_gap(dt_results)
    plot_feature_importances(models, feature_names)
    plot_variable_fragmentation(models, X_train_df, feature_names)
    render_trees(models, feature_names)
    
    run_random_forest(X_train_df, y_train, X_test_df, y_test, dt_results, feature_names, models)
    print("Experimento concluído com sucesso!")

if __name__ == '__main__':
    main()
