import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# Crear directorio para plots si no existe
plots_dir = "python/plots"
os.makedirs(plots_dir, exist_ok=True)

# Datos BKS para las instancias
def get_bks_data():
    bks_data = {
        "p01.dat": 576.87,
        "p02.dat": 473.53,
        "p03.dat": 640.65,
        "p04.dat": 999.21,
        "p05.dat": 750.03,
        "p06.dat": 876.5,
        "p07.dat": 881.97,
        "p08.dat": 4375.49,
        "p09.dat": 3859.17,
        "p10.dat": 3631.11,
        "p11.dat": 3546.06,
        "p12.dat": 1318.95,
        "p13.dat": 1318.95,
        "p14.dat": 1360.12,
        "p15.dat": 2505.42,
        "p16.dat": 2572.23,
        "p17.dat": 2709.09,
        "p18.dat": 3702.85,
        "pr01.dat": 861.32,
        "pr02.dat": 1296.25,
        "pr03.dat": 1803.8,
        "pr04.dat": 2042.45,
        "pr05.dat": 2324.45,
        "pr06.dat": 2663.56,
        "pr07.dat": 1075.12,
        "pr08.dat": 1658.23,
        "pr09.dat": 2131.7,
        "pr10.dat": 2805.53
    }
    return bks_data

# 1. Gráfica de convergencia - Evolución del fitness a lo largo de las generaciones
def plot_convergence_curve(instance_name, convergence_data, bks_value=None, save_dir=plots_dir):
    """Genera una gráfica de la curva de convergencia"""
    plt.figure(figsize=(10, 6))
    
    generations = np.arange(len(convergence_data))
    
    # Calcular mejora porcentual
    initial_fitness = convergence_data[0]
    final_fitness = convergence_data[-1]
    improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
    
    # Gráfica principal
    plt.plot(generations, convergence_data, color='#1f77b4', linewidth=2.5)
    
    # Destacar puntos inicial y final
    plt.scatter([0], [initial_fitness], color='#d62728', s=100, zorder=5, label='Fitness Inicial')
    plt.scatter([len(convergence_data)-1], [final_fitness], color='#2ca02c', s=100, zorder=5, label='Fitness Final')
    
    # Añadir línea horizontal con el BKS si está disponible
    if bks_value:
        plt.axhline(y=bks_value, color='#ff7f0e', linestyle='--', linewidth=1.5, 
                   label=f'BKS: {bks_value:.2f}')
        
        # Calcular gap con BKS
        gap = ((final_fitness - bks_value) / bks_value) * 100
        plt.annotate(f'Gap: {gap:.2f}%', 
                    xy=(len(convergence_data)*0.85, bks_value*1.05),
                    fontsize=12, color='#ff7f0e')
    
    # Añadir información de mejora
    plt.annotate(f'Mejora: {improvement:.2f}%', 
                xy=(len(convergence_data)*0.7, initial_fitness*0.95),
                fontsize=12, color='#1f77b4')
    
    # Configurar ejes y etiquetas
    plt.title(f'Convergencia del Algoritmo - Instancia {instance_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Distancia Total)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Usar escala logarítmica en Y si la mejora es grande
    if improvement > 50:
        plt.yscale('log')
        plt.ylabel('Fitness (Escala Logarítmica)', fontsize=12)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Guardar la imagen
    output_path = os.path.join(save_dir, f'convergencia_{instance_name.replace(".", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de convergencia guardada en: {output_path}")
    plt.close()
    
    return output_path

# 2. Gráfica comparativa de fitness vs BKS
def plot_fitness_vs_bks(results_data, save_dir=plots_dir):
    """Genera un gráfico de barras comparando los resultados obtenidos con los BKS"""
    bks_data = get_bks_data()
    
    # Preparar datos para gráfica
    plot_data = []
    for instance, data in results_data.items():
        if instance in bks_data:
            plot_data.append({
                'instance': instance.replace('.dat', ''),
                'fitness': data['fitness'],
                'bks': bks_data[instance],
                'gap': ((data['fitness'] - bks_data[instance]) / bks_data[instance]) * 100
            })
    
    # Ordenar por gap
    plot_data = sorted(plot_data, key=lambda x: x['gap'])
    
    # Limitar a 10 instancias para mejor visualización si hay más
    if len(plot_data) > 10:
        # Tomar 5 con menor gap y 5 con mayor gap
        plot_data = plot_data[:5] + plot_data[-5:]
    
    # Crear DataFrame
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 7))
    
    # Graficar barras agrupadas
    x = np.arange(len(df))
    width = 0.35
    
    # Normalizar por BKS para mejor comparación
    ax = plt.gca()
    bars1 = ax.bar(x - width/2, df['fitness'] / df['bks'], width, label='Fitness Obtenido / BKS', color='#1f77b4')
    bars2 = ax.bar(x + width/2, df['bks'] / df['bks'], width, label='BKS (Normalizado)', color='#2ca02c')
    
    # Añadir gap en las barras
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.annotate(f"{df['gap'].iloc[i]:.1f}%",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, rotation=90)
    
    # Configurar ejes y etiquetas
    ax.set_ylabel('Proporción respecto a BKS', fontsize=12)
    ax.set_title('Comparación de Fitness obtenido vs. BKS', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['instance'], rotation=45, ha='right')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Añadir gap promedio
    avg_gap = df['gap'].mean()
    plt.figtext(0.5, 0.01, f"Gap promedio: {avg_gap:.2f}%", ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Guardar la imagen
    output_path = os.path.join(save_dir, 'comparacion_fitness_bks.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de comparación vs BKS guardada en: {output_path}")
    plt.close()
    
    return output_path

# 3. Gráfica de tiempo de ejecución vs complejidad de la instancia
def plot_execution_time_vs_complexity(results_data, save_dir=plots_dir):
    """Genera una gráfica relacionando el tiempo de ejecución con la complejidad de la instancia"""
    # Preparar datos para gráfica
    plot_data = []
    for instance, data in results_data.items():
        # Contar número de rutas como indicador de complejidad
        routes_count = len(data.get('best_solution', []))
        
        if routes_count > 0:
            plot_data.append({
                'instance': instance.replace('.dat', ''),
                'execution_time': data['execution_time'],
                'routes': routes_count,
                # Calcular complejidad como producto de rutas y generaciones
                'complexity': routes_count * data.get('generation', 1)
            })
    
    # Crear DataFrame
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    
    # Gráfica de dispersión con tamaño proporcional al número de rutas
    sizes = df['routes'] * 20
    scatter = plt.scatter(df['routes'], df['execution_time'], s=sizes, 
                         alpha=0.6, c=df['execution_time'], cmap='viridis')
    
    # Añadir etiquetas a los puntos
    for i, row in df.iterrows():
        plt.annotate(row['instance'], 
                    (row['routes'], row['execution_time']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Ajustar una curva de tendencia
    z = np.polyfit(df['routes'], df['execution_time'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df['routes'].min(), df['routes'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.7, label="Tendencia")
    
    plt.colorbar(scatter, label='Tiempo de ejecución (s)')
    
    # Configurar ejes y etiquetas
    plt.title('Tiempo de Ejecución vs. Complejidad de Instancia', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Rutas', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar la imagen
    output_path = os.path.join(save_dir, 'tiempo_vs_complejidad.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de tiempo vs complejidad guardada en: {output_path}")
    plt.close()
    
    return output_path

# 4. Gráfica de clasificación de gaps
def plot_gap_classification(results_data, save_dir=plots_dir):
    """Genera una gráfica clasificando los gaps en diferentes categorías"""
    bks_data = get_bks_data()
    
    # Preparar datos para gráfica
    plot_data = []
    for instance, data in results_data.items():
        if instance in bks_data:
            gap = ((data['fitness'] - bks_data[instance]) / bks_data[instance]) * 100
            plot_data.append({
                'instance': instance.replace('.dat', ''),
                'fitness': data['fitness'],
                'bks': bks_data[instance],
                'gap': gap
            })
    
    # Ordenar por gap
    plot_data = sorted(plot_data, key=lambda x: x['gap'])
    
    # Crear DataFrame
    df = pd.DataFrame(plot_data)
    
    # Definir categorías de gap
    categories = [
        {'name': 'Excelente', 'range': (0, 10), 'color': '#2ca02c'},
        {'name': 'Bueno', 'range': (10, 25), 'color': '#1f77b4'},
        {'name': 'Regular', 'range': (25, 50), 'color': '#ff7f0e'},
        {'name': 'Alto', 'range': (50, 100), 'color': '#d62728'},
        {'name': 'Muy Alto', 'range': (100, float('inf')), 'color': '#7f7f7f'}
    ]
    
    # Asignar color según categoría
    colors = []
    categories_count = {cat['name']: 0 for cat in categories}
    
    for gap in df['gap']:
        for cat in categories:
            if cat['range'][0] <= gap < cat['range'][1]:
                colors.append(cat['color'])
                categories_count[cat['name']] += 1
                break
        else:
            # Si no entra en ninguna categoría (por si acaso)
            colors.append('#7f7f7f')
    
    plt.figure(figsize=(12, 7))
    
    # Crear gráfica de barras
    bars = plt.bar(df['instance'], df['gap'], color=colors)
    
    # Añadir valor de gap encima de cada barra
    for bar, gap in zip(bars, df['gap']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{gap:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Configurar ejes y etiquetas
    plt.title('Clasificación de Gaps por Instancia', fontsize=14, fontweight='bold')
    plt.xlabel('Instancia', fontsize=12)
    plt.ylabel('Gap (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=90)
    
    # Añadir leyenda con las categorías
    legend_elements = [Patch(facecolor=cat['color'], label=f"{cat['name']} ({cat['range'][0]}-{cat['range'][1]}%): {categories_count[cat['name']]}") 
                      for cat in categories]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Guardar la imagen
    output_path = os.path.join(save_dir, 'clasificacion_gaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de clasificación de gaps guardada en: {output_path}")
    plt.close()
    
    return output_path

# 5. Gráfica de mejora acumulada durante la convergencia
def plot_accumulated_improvement(results_data, save_dir=plots_dir):
    """Genera una gráfica de mejora acumulada durante la convergencia"""
    # Seleccionar una instancia representativa o la primera disponible
    selected_instance = None
    for instance, data in results_data.items():
        if 'convergence_history' in data and len(data['convergence_history']) > 10:
            selected_instance = (instance, data)
            break
    
    if not selected_instance:
        print("No se encontraron datos de convergencia adecuados")
        return None
    
    instance_name, instance_data = selected_instance
    convergence = instance_data['convergence_history']
    
    plt.figure(figsize=(10, 6))
    
    # Valores de fitness
    generations = np.arange(len(convergence))
    
    # Calcular mejora acumulada en porcentaje
    initial_fitness = convergence[0]
    accumulated_improvement = [(initial_fitness - val) / initial_fitness * 100 for val in convergence]
    
    # Gráfica de mejora acumulada
    plt.plot(generations, accumulated_improvement, color='#1f77b4', linewidth=2.5)
    
    # Dividir la convergencia en fases
    n_phases = 3
    phase_length = len(generations) // n_phases
    
    # Destacar fases de convergencia
    phases = []
    for i in range(n_phases):
        start_idx = i * phase_length
        end_idx = (i+1) * phase_length if i < n_phases-1 else len(generations)-1
        
        # Calcular la mejora en esta fase
        if i == 0:
            improvement = accumulated_improvement[end_idx]
        else:
            improvement = accumulated_improvement[end_idx] - accumulated_improvement[start_idx-1]
        
        phases.append({
            'start': start_idx,
            'end': end_idx,
            'improvement': improvement
        })
    
    # Sombrear las fases
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Fase inicial', 'Fase intermedia', 'Fase final']
    
    for i, phase in enumerate(phases):
        plt.axvspan(phase['start'], phase['end'], alpha=0.2, color=colors[i])
        plt.annotate(f"{labels[i]}: {phase['improvement']:.2f}%", 
                    xy=((phase['start'] + phase['end'])/2, 
                       accumulated_improvement[int((phase['start'] + phase['end'])/2)]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', fontsize=10, color=colors[i])
    
    # Configurar ejes y etiquetas
    plt.title(f'Mejora Acumulada Durante la Convergencia - {instance_name}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Mejora Acumulada (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Añadir mejora total
    plt.annotate(f"Mejora total: {accumulated_improvement[-1]:.2f}%", 
                xy=(len(generations)*0.8, accumulated_improvement[-1]),
                fontsize=12, color='#d62728', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar la imagen
    output_path = os.path.join(save_dir, 'mejora_acumulada.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de mejora acumulada guardada en: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Función principal que lee el archivo JSON y genera todas las gráficas"""
    # Configurar estilo de gráficas
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # Leer el archivo JSON
    json_file = 'resultados.json'
    try:
        with open(json_file, 'r') as f:
            results_data = json.load(f)
        print(f"Archivo JSON cargado correctamente: {json_file}")
    except Exception as e:
        print(f"Error al cargar el archivo JSON: {e}")
        return
    
    # Obtener los datos BKS
    bks_data = get_bks_data()
    
    # 1. Generar gráficas de convergencia para cada instancia
    for instance, data in results_data.items():
        if 'convergence_history' in data and len(data['convergence_history']) > 0:
            bks = bks_data.get(instance)
            print(f"Generando gráfica de convergencia para {instance}...")
            plot_convergence_curve(instance, data['convergence_history'], bks)
    
    # 2. Generar gráfica comparativa con BKS
    print("Generando gráfica comparativa con BKS...")
    plot_fitness_vs_bks(results_data)
    
    # 3. Generar gráfica de tiempo vs complejidad
    print("Generando gráfica de tiempo vs complejidad...")
    plot_execution_time_vs_complexity(results_data)
    
    # 4. Generar gráfica de clasificación de gaps
    print("Generando gráfica de clasificación de gaps...")
    plot_gap_classification(results_data)
    
    # 5. Generar gráfica de mejora acumulada
    print("Generando gráfica de mejora acumulada...")
    plot_accumulated_improvement(results_data)
    
    print("\nTodas las gráficas han sido generadas en el directorio:", plots_dir)

if __name__ == "__main__":
    main()