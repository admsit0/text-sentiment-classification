"""
Práctica 2 - Ejercicio 5: Evaluación de algoritmos de clasificación y elaboración de informe
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid
"""

import json
import numpy as np
import pickle
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evalúa modelos de clasificación y genera informes"""
    
    def __init__(self):
        """Inicializa el evaluador"""
        self.results = {}
        
    def load_test_data(self, dataset_dir):
        """
        Carga datos de test
        """
        print(f"    [LOGGER] Cargando 'test_X.npy' y 'test_y.npy' desde {dataset_dir}")
        X_test = np.load(os.path.join(dataset_dir, 'test_X.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'test_y.npy'))
        
        # --- CORRECCIÓN ETIQUETAS: Convertir etiquetas a numérico ---
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if not np.issubdtype(y_test.dtype, np.number):
            print("    [LOGGER] ℹ Convirtiendo etiquetas de test a numérico (0=neg, 1=neu, 2=pos)...")
            y_test = np.vectorize(label_map.get)(y_test)
            
        return X_test, y_test
    
    def load_model(self, model_file):
        """
        Carga un modelo entrenado
        """
        print(f"      [LOGGER] Cargando modelo desde {model_file}...")
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    
    def evaluate_model(self, model_info, X_test, y_test):
        """
        Evalúa un modelo en el conjunto de test
        """
        model = model_info['best_estimator']
        scaler = model_info.get('scaler', None)
        
        X_test_proc = X_test
        if scaler is not None:
            print("      [LOGGER] Aplicando scaler a los datos de test...")
            # --- CORRECCIÓN MEMORIA: Forzar float32 ---
            X_test_proc = scaler.transform(X_test).astype(np.float32)
        
        print("      [LOGGER] Realizando predicciones en el conjunto de test...")
        y_pred = model.predict(X_test_proc)
        
        print("      [LOGGER] Calculando métricas (Accuracy, Precision, Recall, F1)...")
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # --- CORRECCIÓN ETIQUETAS: Usar etiquetas numéricas ---
        labels_numeric = np.array([0, 1, 2])
        unique_labels = np.unique(y_test)
        # Usar las etiquetas que realmente están en y_test para el reporte
        present_labels = [l for l in labels_numeric if l in unique_labels]

        precision_per_class = precision_score(y_test, y_pred, average=None, labels=present_labels, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, labels=present_labels, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=present_labels, zero_division=0)
        
        print("      [LOGGER] Generando matriz de confusión...")
        cm = confusion_matrix(y_test, y_pred, labels=labels_numeric)
        
        report = classification_report(y_test, y_pred, labels=present_labels, target_names=['negative', 'neutral', 'positive'], output_dict=True, zero_division=0)
        
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_per_class': {
                # Mapear 0, 1, 2 a 'negative', 'neutral', 'positive' para el JSON
                str(label): float(score) 
                for label, score in zip(['negative', 'neutral', 'positive'], precision_per_class)
            },
            'recall_per_class': {
                str(label): float(score)
                for label, score in zip(['negative', 'neutral', 'positive'], recall_per_class)
            },
            'f1_per_class': {
                str(label): float(score)
                for label, score in zip(['negative', 'neutral', 'positive'], f1_per_class)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, labels, title, output_file):
        """
        Genera gráfica de matriz de confusión
        """
        print(f"      [LOGGER] Guardando gráfica de Matriz de Confusión en {output_file}...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Frecuencia'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Etiqueta Verdadera', fontsize=12)
        plt.xlabel('Etiqueta Predicha', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_table(self, all_results):
        """
        Crea tabla comparativa de todos los modelos
        """
        rows = []
        
        for rep_name, models in all_results.items():
            for model_name, result in models.items():
                if 'error' in result or 'status' in result:
                    continue
                
                row = {
                    'Representación': rep_name,
                    'Modelo': result['display_name'],
                    'Accuracy': result['test_metrics']['accuracy'],
                    'Precision (macro)': result['test_metrics']['precision_macro'],
                    'Recall (macro)': result['test_metrics']['recall_macro'],
                    'F1-score (macro)': result['test_metrics']['f1_macro'],
                    'CV Score': result['cv_score'],
                    'Val Score': result.get('val_score', np.nan)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ordenar por F1-score descendente
        df = df.sort_values('F1-score (macro)', ascending=False)
        
        return df
    
    def plot_model_comparison(self, df, output_file):
        """
        Genera gráfica comparativa de modelos
        """
        print(f"  [LOGGER] Guardando gráfica de Comparación de Modelos en {output_file}...")
        df_top = df.head(10).copy()
        df_top['Model_Rep'] = df_top['Modelo'] + '\n(' + df_top['Representación'].str.replace('_', ' ') + ')'
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_top))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, df_top[metric], width, label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Modelo (Representación)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Top 10 Modelos por F1-score', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_top['Model_Rep'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_representation_comparison(self, df, output_file):
        """
        Compara rendimiento por tipo de representación
        """
        print(f"  [LOGGER] Guardando gráfica de Comparación de Representaciones en {output_file}...")
        rep_stats = df.groupby('Representación').agg({
            'Accuracy': 'mean',
            'Precision (macro)': 'mean',
            'Recall (macro)': 'mean',
            'F1-score (macro)': 'mean'
        }).reset_index()
        
        rep_stats = rep_stats.sort_values('F1-score (macro)', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(rep_stats))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, rep_stats[metric], width, label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Representación', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score Promedio', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Representaciones (Promedio de Modelos)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace('_', '\n') for r in rep_stats['Representación']], 
            rotation=45, ha='right', fontsize=9
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_latex_table(self, df, output_file, top_n=10):
        """
        Genera tabla en formato LaTeX
        """
        print(f"  [LOGGER] Guardando tabla LaTeX en {output_file}...")
        df_top = df.head(top_n).copy()
        
        # Formatear valores
        for col in ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']:
            df_top[col] = df_top[col].apply(lambda x: f"{x:.4f}")
        
        # Crear tabla LaTeX
        latex = df_top.to_latex(
            index=False,
            column_format='l|l|c|c|c|c',
            caption=f'Top {top_n} modelos por F1-score en conjunto de test',
            label='tab:top_models'
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex)


def evaluate_all_models(datasets_dir, models_dir, output_dir):
    """
    Evalúa todos los modelos entrenados
    """
    print("="*60)
    print("EVALUACIÓN DE MODELOS EN CONJUNTO DE TEST (E5)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    # Listar representaciones
    representations = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    
    print(f"\n[LOGGER] Representaciones encontradas en '{models_dir}': {len(representations)}")
    
    # Evaluar cada representación
    for i, rep_name in enumerate(representations, 1):
        print(f"\n{'='*60}")
        print(f"REPRESENTACIÓN {i}/{len(representations)}: {rep_name}")
        print(f"{'='*60}")
        
        rep_models_dir = os.path.join(models_dir, rep_name)
        rep_datasets_dir = os.path.join(datasets_dir, rep_name)
        rep_output_dir = os.path.join(output_dir, rep_name)
        os.makedirs(rep_output_dir, exist_ok=True)
        
        # Cargar datos de test
        print(f"\n  [LOGGER] Cargando datos de test para {rep_name}...")
        try:
            X_test, y_test = evaluator.load_test_data(rep_datasets_dir)
            print(f"    [LOGGER] Datos de test cargados: {X_test.shape}, {len(y_test)} etiquetas")
        except Exception as e:
            print(f"    ✗ [LOGGER] Error cargando datos: {e}")
            continue
        
        # Evaluar cada modelo
        model_files = [f for f in os.listdir(rep_models_dir) if f.endswith('.pkl')]
        print(f"\n  [LOGGER] Modelos a evaluar: {len(model_files)}")
        
        rep_results = {}
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            model_path = os.path.join(rep_models_dir, model_file)
            
            print(f"\n    [LOGGER] Evaluando: {model_name}")
            
            try:
                # Cargar modelo
                model_info = evaluator.load_model(model_path)
                
                # Evaluar
                test_metrics = evaluator.evaluate_model(model_info, X_test, y_test)
                
                # Guardar resultados
                rep_results[model_name] = {
                    'display_name': model_info.get('display_name', model_name),
                    'best_params': model_info.get('best_params', {}),
                    'cv_score': model_info.get('best_cv_score'),
                    'val_score': model_info.get('val_score'),
                    'test_metrics': test_metrics
                }
                
                # Mostrar métricas principales
                print(f"      [LOGGER] Accuracy:  {test_metrics['accuracy']:.4f}")
                print(f"      [LOGGER] Precision: {test_metrics['precision_macro']:.4f}")
                print(f"      [LOGGER] Recall:    {test_metrics['recall_macro']:.4f}")
                print(f"      [LOGGER] F1-score:  {test_metrics['f1_macro']:.4f}")
                
                # Generar matriz de confusión
                labels = ['negative', 'neutral', 'positive']
                cm = np.array(test_metrics['confusion_matrix'])
                cm_file = os.path.join(rep_output_dir, f'{model_name}_confusion_matrix.png')
                evaluator.plot_confusion_matrix(
                    cm, labels,
                    f'{model_info.get("display_name", model_name)}\n{rep_name}',
                    cm_file
                )
                
            except Exception as e:
                print(f"      [LOGGER] ✗ Error: {e}")
                rep_results[model_name] = {'error': str(e)}
        
        all_results[rep_name] = rep_results
        
        # Guardar resultados de la representación
        rep_results_file = os.path.join(rep_output_dir, 'evaluation_results.json')
        print(f"\n  [LOGGER] Guardando resultados de {rep_name} en {rep_results_file}...")
        with open(rep_results_file, 'w', encoding='utf-8') as f:
            json.dump(rep_results, f, indent=2)
        print(f"  ✓ [LOGGER] Resultados guardados: {rep_results_file}")
    
    return all_results


def generate_report(all_results, output_dir, datasets_dir):
    """
    Genera informe completo de evaluación
    """
    print("\n" + "="*60)
    print("GENERACIÓN DE INFORME (E5)")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    # Crear tabla comparativa
    print("\n[LOGGER] Generando tabla comparativa...")
    df = evaluator.create_comparison_table(all_results)
    
    # Guardar tabla en CSV
    csv_file = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Tabla CSV: {csv_file}")
    
    # Guardar tabla en LaTeX
    latex_file = os.path.join(output_dir, 'comparison_table.tex')
    evaluator.generate_latex_table(df, latex_file)
    print(f"  ✓ Tabla LaTeX: {latex_file}")
    
    # Generar gráficas
    print("\n[LOGGER] Generando gráficas comparativas...")
    
    # Comparación de modelos
    models_plot = os.path.join(output_dir, 'models_comparison.png')
    evaluator.plot_model_comparison(df, models_plot)
    print(f"  ✓ Comparación de modelos: {models_plot}")
    
    # Comparación de representaciones
    reps_plot = os.path.join(output_dir, 'representations_comparison.png')
    evaluator.plot_representation_comparison(df, reps_plot)
    print(f"  ✓ Comparación de representaciones: {reps_plot}")
    
    # Identificar mejor modelo
    if df.empty:
        print("\n✗ [LOGGER] ¡ERROR! No se encontraron resultados válidos para generar el informe.")
        return
        
    best_model_row = df.iloc[0]
    print("\n" + "="*60)
    print("MEJOR MODELO")
    print("="*60)
    print(f"\n[LOGGER] Modelo: {best_model_row['Modelo']}")
    print(f"[LOGGER] Representación: {best_model_row['Representación']}")
    print(f"[LOGGER] F1-score (macro): {best_model_row['F1-score (macro)']:.4f}")
    print(f"[LOGGER] Accuracy: {best_model_row['Accuracy']:.4f}")
    print(f"[LOGGER] Precision (macro): {best_model_row['Precision (macro)']:.4f}")
    print(f"[LOGGER] Recall (macro): {best_model_row['Recall (macro)']:.4f}")
    
    # Generar resumen ejecutivo
    print("\n[LOGGER] Generando resumen ejecutivo...")
    summary = {
        'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_representaciones': len(all_results),
        'num_modelos_totales': len(df),
        'mejor_modelo': {
            'modelo': best_model_row['Modelo'],
            'representacion': best_model_row['Representación'],
            'metricas': {
                'accuracy': float(best_model_row['Accuracy']),
                'precision_macro': float(best_model_row['Precision (macro)']),
                'recall_macro': float(best_model_row['Recall (macro)']),
                'f1_macro': float(best_model_row['F1-score (macro)'])
            }
        },
        'top_5_modelos': []
    }
    
    for idx, row in df.head(5).iterrows():
        summary['top_5_modelos'].append({
            'rank': len(summary['top_5_modelos']) + 1,
            'modelo': row['Modelo'],
            'representacion': row['Representación'],
            'f1_score': float(row['F1-score (macro)'])
        })
    
    summary_file = os.path.join(output_dir, 'executive_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Resumen ejecutivo: {summary_file}")
    
    # Cargar configuración de etiquetado
    labeling_config_file = os.path.join(datasets_dir, 'labeling_config.json')
    if os.path.exists(labeling_config_file):
        with open(labeling_config_file, 'r') as f:
            labeling_config = json.load(f)
    else:
        labeling_config = {}
    
    # Generar informe en Markdown
    print("\n[LOGGER] Generando informe en Markdown...")
    markdown_file = os.path.join(output_dir, 'informe_resultados.md')
    generate_markdown_report(df, best_model_row, all_results, labeling_config, markdown_file)
    print(f"  ✓ Informe Markdown: {markdown_file}")


def generate_markdown_report(df, best_model, all_results, labeling_config, output_file):
    """
    Genera informe en formato Markdown
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Informe de Resultados - Práctica 2\n")
        f.write("## Clasificación de Texto para Análisis de Sentimiento\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("---\n\n")
        
        # Configuración de etiquetado
        f.write("## 1. Configuración del Experimento\n\n")
        f.write("### Etiquetado de Clases\n\n")
        if labeling_config:
            f.write("Umbrales de rating utilizados:\n\n")
            f.write(f"- **Positiva**: Rating ≥ {labeling_config.get('positive_threshold', 'N/A')}\n")
            f.write(f"- **Neutra**: {labeling_config.get('neutral_min', 'N/A')} ≤ Rating ≤ {labeling_config.get('neutral_max', 'N/A')}\n")
            f.write(f"- **Negativa**: Rating ≤ {labeling_config.get('negative_threshold', 'N/A')}\n\n")
            
            if 'distribution' in labeling_config:
                dist = labeling_config['distribution']
                f.write("Distribución de clases (antes del balanceo):\n\n")
                f.write(f"- Positivas: {dist.get('positive', 'N/A')}\n")
                f.write(f"- Neutras: {dist.get('neutral', 'N/A')}\n")
                f.write(f"- Negativas: {dist.get('negative', 'N/A')}\n\n")
        
        # Representaciones
        f.write("### Representaciones Vectoriales\n\n")
        f.write(f"Total de representaciones evaluadas: {len(all_results)}\n\n")
        for rep_name in all_results.keys():
            f.write(f"- {rep_name}\n")
        f.write("\n")
        
        # Modelos
        f.write("### Modelos de Clasificación\n\n")
        modelos_unicos = df['Modelo'].unique()
        f.write(f"Total de modelos evaluados: {len(modelos_unicos)}\n\n")
        for modelo in modelos_unicos:
            f.write(f"- {modelo}\n")
        f.write("\n---\n\n")
        
        # Mejor modelo
        f.write("## 2. Mejor Modelo\n\n")
        f.write(f"**Modelo:** {best_model['Modelo']}\n\n")
        f.write(f"**Representación:** {best_model['Representación']}\n\n")
        f.write("### Métricas de Rendimiento\n\n")
        f.write("| Métrica | Valor |\n")
        f.write("|---------|-------|\n")
        f.write(f"| Accuracy | {best_model['Accuracy']:.4f} |\n")
        f.write(f"| Precision (macro) | {best_model['Precision (macro)']:.4f} |\n")
        f.write(f"| Recall (macro) | {best_model['Recall (macro)']:.4f} |\n")
        f.write(f"| F1-score (macro) | {best_model['F1-score (macro)']:.4f} |\n")
        f.write(f"| CV Score | {best_model['CV Score']:.4f} |\n")
        if not pd.isna(best_model['Val Score']):
            f.write(f"| Val Score | {best_model['Val Score']:.4f} |\n")
        f.write("\n---\n\n")
        
        # Top 10 modelos
        f.write("## 3. Top 10 Modelos\n\n")
        f.write("| Rank | Modelo | Representación | F1-score | Accuracy |\n")
        f.write("|------|--------|----------------|----------|----------|\n")
        for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
            f.write(f"| {idx} | {row['Modelo']} | {row['Representación']} | ")
            f.write(f"{row['F1-score (macro)']:.4f} | {row['Accuracy']:.4f} |\n")
        f.write("\n---\n\n")
        
        # Comparación de representaciones
        f.write("## 4. Análisis de Representaciones\n\n")
        rep_stats = df.groupby('Representación').agg({
            'F1-score (macro)': ['mean', 'std', 'max']
        }).round(4)
        rep_stats.columns = ['F1 Promedio', 'F1 Std', 'F1 Máximo']
        rep_stats = rep_stats.sort_values('F1 Promedio', ascending=False)
        
        f.write(rep_stats.to_markdown())
        f.write("\n\n---\n\n")
        
        # Análisis de modelos
        f.write("## 5. Análisis de Modelos\n\n")
        model_stats = df.groupby('Modelo').agg({
            'F1-score (macro)': ['mean', 'std', 'max']
        }).round(4)
        model_stats.columns = ['F1 Promedio', 'F1 Std', 'F1 Máximo']
        model_stats = model_stats.sort_values('F1 Promedio', ascending=False)
        
        f.write(model_stats.to_markdown())
        f.write("\n\n---\n\n")
        
        # Conclusiones
        f.write("## 6. Conclusiones\n\n")
        f.write(f"- El mejor modelo obtenido fue **{best_model['Modelo']}** ")
        f.write(f"con la representación **{best_model['Representación']}**, ")
        f.write(f"alcanzando un F1-score (macro) de **{best_model['F1-score (macro)']:.4f}**.\n\n")
        
        best_rep = rep_stats.index[0]
        f.write(f"- La representación que mejor funcionó en promedio fue **{best_rep}** ")
        f.write(f"con un F1-score promedio de **{rep_stats.loc[best_rep, 'F1 Promedio']:.4f}**.\n\n")
        
        best_model_type = model_stats.index[0]
        f.write(f"- El tipo de modelo más efectivo fue **{best_model_type}** ")
        f.write(f"con un F1-score promedio de **{model_stats.loc[best_model_type, 'F1 Promedio']:.4f}**.\n\n")


def main(datasets_dir, models_dir, output_dir):
    """
    Función principal de evaluación
    """
    print("="*60)
    print("EVALUACIÓN Y GENERACIÓN DE INFORME - EJERCICIO 5")
    print("="*60)
    
    # Evaluar todos los modelos
    all_results = evaluate_all_models(datasets_dir, models_dir, output_dir)
    
    # Guardar todos los resultados
    all_results_file = os.path.join(output_dir, 'all_results.json')
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ [LOGGER] Todos los resultados guardados: {all_results_file}")
    
    # Generar informe completo
    generate_report(all_results, output_dir, datasets_dir)
    
    print("\n" + "="*60)
    print("¡EVALUACIÓN (E5) COMPLETADA!")
    print("="*60)
    print(f"\n[LOGGER] Resultados guardados en: {output_dir}")
    print("\n[LOGGER] Archivos generados:")
    print("  • all_results.json - Resultados completos")
    print("  • comparison_table.csv - Tabla comparativa")
    print("  • comparison_table.tex - Tabla en LaTeX")
    print("  • models_comparison.png - Gráfica de modelos")
    print("  • representations_comparison.png - Gráfica de representaciones")
    print("  • executive_summary.json - Resumen ejecutivo")
    print("  • informe_resultados.md - Informe en Markdown")
    print("  • <representacion>/ - Resultados por representación")
    print("      └── <modelo>_confusion_matrix.png - Matrices de confusión")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evalúa modelos y genera informe'
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='datasets',
        help='Directorio con datasets del Ejercicio 3'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='modelos',
        help='Directorio con modelos del Ejercicio 4'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluacion',
        help='Directorio de salida para resultados'
    )
    
    args = parser.parse_args()
    
    main(
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )