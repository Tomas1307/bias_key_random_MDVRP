import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import re
import json
from sklearn.decomposition import PCA  # Se utiliza para reducir dimensiones

def load_results_from_files(results_dir):
    """
    Carga los resultados de hiperparámetros desde archivos de texto.
    Formato esperado:
    config_string, best_distance, convergence_gen, execution_time, num_routes, num_used_depots
    
    Args:
        results_dir: Directorio donde se encuentran los archivos de resultados

    Returns:
        DataFrame: DataFrame con los resultados procesados
    """
    all_results = []
    
    # Recorrer todos los archivos en el directorio
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            instance_name = os.path.splitext(filename)[0]
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        # Ignorar líneas de comentario
                        if line.startswith('#') or not line.strip():
                            continue
                        
                        # Parsear la línea
                        parts = line.strip().split(',')
                        if len(parts) < 6:
                            print(f"Advertencia: Formato incorrecto en línea: {line}")
                            continue
                        
                        config_string = parts[0].strip()
                        best_distance = float(parts[1].strip())
                        convergence_gen = int(parts[2].strip())
                        execution_time = float(parts[3].strip())
                        num_routes = int(parts[4].strip())
                        num_used_depots = int(parts[5].strip())
                        
                        # Extraer parámetros del config_string
                        # Formato esperado: pop=X_elite=Y_mutant=Z_gen=W_bias=V
                        params = {}
                        param_parts = config_string.split('_')
                        for param in param_parts:
                            if '=' in param:
                                key, value = param.split('=')
                                if key == 'pop':
                                    params['population_size'] = int(value)
                                elif key == 'elite':
                                    params['elite_percent'] = float(value)
                                elif key == 'mutant':
                                    params['mutants_percent'] = float(value)
                                elif key == 'gen':
                                    params['max_generations'] = int(value)
                                elif key == 'bias':
                                    params['p_bias'] = float(value)
                        
                        # Añadir resultado al conjunto
                        result = {
                            'instance': instance_name,
                            'fitness': best_distance,
                            'convergence_gen': convergence_gen,
                            'execution_time': execution_time,
                            'num_routes': num_routes,
                            'num_used_depots': num_used_depots,
                            'is_valid': True  # Asumimos que todos los resultados son válidos
                        }
                        result.update(params)
                        all_results.append(result)
            
            except Exception as e:
                print(f"Error al procesar archivo {filename}: {str(e)}")
    
    # Convertir a DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        print("No se encontraron resultados para procesar.")
        return pd.DataFrame()

def generate_visualizations(df_results, results_dir):
    """
    Genera visualizaciones para analizar los resultados de la optimización.

    Args:
        df_results: DataFrame con resultados
        results_dir: Directorio donde guardar las visualizaciones
    """
    plots_dir = os.path.join(results_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 1. Resumen por instancia
    instances = df_results['instance'].unique()

    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]

        # Encontrar la mejor configuración
        best_config = instance_data.loc[instance_data['fitness'].idxmin()]

        print(f"\nMejor configuración para {instance}:")
        print(f"  Population Size: {best_config['population_size']}")
        print(f"  Elite Percent: {best_config['elite_percent']}")
        print(f"  Mutants Percent: {best_config['mutants_percent']}")
        print(f"  Max Generations: {best_config['max_generations']}")
        print(f"  P-Bias: {best_config['p_bias']}")
        print(f"  Fitness: {best_config['fitness']}")
        print(f"  Tiempo: {best_config['execution_time']:.2f}s")

    # 2. Gráfico 3D: Visualización tipo cubo para 3 hiperparámetros
    param_combinations = [
        ('population_size', 'elite_percent', 'mutants_percent'),
        ('population_size', 'elite_percent', 'p_bias'),
        ('population_size', 'max_generations', 'p_bias'),
        ('elite_percent', 'mutants_percent', 'p_bias')
    ]

    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]

        for param_x, param_y, param_z in param_combinations:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Agrupar y obtener el mejor fitness para cada combinación
            unique_points = instance_data.groupby([param_x, param_y, param_z])['fitness'].min().reset_index()

            # Normalización de colores
            norm = plt.Normalize(unique_points['fitness'].min(), unique_points['fitness'].max())
            colors = plt.cm.viridis(norm(unique_points['fitness']))

            # Tamaño de punto basado en fitness (mejor = mayor tamaño)
            sizes = 100 - 90 * (unique_points['fitness'] - unique_points['fitness'].min()) / (
                unique_points['fitness'].max() - unique_points['fitness'].min())
            sizes = sizes.clip(lower=20)

            scatter = ax.scatter(
                unique_points[param_x],
                unique_points[param_y],
                unique_points[param_z],
                c=unique_points['fitness'],
                cmap='viridis',
                s=sizes,
                alpha=0.7
            )

            cbar = plt.colorbar(scatter)
            cbar.set_label('Fitness (menor es mejor)')

            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_zlabel(param_z)
            plt.title(f'Optimización de Hiperparámetros para {instance}\n({param_x} vs {param_y} vs {param_z})')

            filename = f"{instance}_{param_x}_{param_y}_{param_z}_3d.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

    # 3. Gráficos de calor (heatmaps) para análisis 2D
    param_pairs = [
        ('population_size', 'elite_percent'),
        ('population_size', 'mutants_percent'),
        ('elite_percent', 'mutants_percent'),
        ('max_generations', 'p_bias'),
        ('population_size', 'p_bias')
    ]

    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]

        for param_x, param_y in param_pairs:
            heatmap_data = instance_data.pivot_table(
                values='fitness',
                index=param_y,
                columns=param_x,
                aggfunc='mean'
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis_r")
            plt.title(f'Mapa de calor para {instance}: {param_y} vs {param_x}')

            filename = f"{instance}_{param_x}_{param_y}_heatmap.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

    # 4. Gráfico resumen de impacto de cada hiperparámetro
    params = ['population_size', 'elite_percent', 'mutants_percent', 'max_generations', 'p_bias']

    for param in params:
        plt.figure(figsize=(12, 6))
        param_performance = df_results.groupby(param)['fitness'].mean().reset_index()
        sns.barplot(x=param, y='fitness', data=param_performance, palette='viridis')
        plt.title(f'Impacto de {param} en el Fitness')
        plt.ylabel('Fitness promedio (menor es mejor)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'impact_{param}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Tiempo de ejecución vs población
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='population_size', y='execution_time', hue='instance', data=df_results, marker='o')
    plt.title('Tiempo de ejecución vs Tamaño de población')
    plt.xlabel('Tamaño de población')
    plt.ylabel('Tiempo (s)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'time_vs_population.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Mejor configuración general
    best_configs = pd.DataFrame()
    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]
        min_fitness = instance_data['fitness'].min()
        max_fitness = instance_data['fitness'].max()
        if max_fitness > min_fitness:
            instance_data['norm_fitness'] = (instance_data['fitness'] - min_fitness) / (max_fitness - min_fitness)
        else:
            instance_data['norm_fitness'] = 0
        best_configs = pd.concat([best_configs, instance_data])

    avg_perf = best_configs.groupby(params)['norm_fitness'].mean().reset_index()
    best_overall = avg_perf.loc[avg_perf['norm_fitness'].idxmin()]

    print("\nMejor configuración general:")
    for param in params:
        print(f"  {param}: {best_overall[param]}")
    print(f"  Rendimiento normalizado: {best_overall['norm_fitness']:.4f}")

    best_config_file = os.path.join(results_dir, 'best_configuration.json')
    with open(best_config_file, 'w') as f:
        json.dump({
            'population_size': int(best_overall['population_size']),
            'elite_percent': float(best_overall['elite_percent']),
            'mutants_percent': float(best_overall['mutants_percent']),
            'max_generations': int(best_overall['max_generations']),
            'p_bias': float(best_overall['p_bias'])
        }, f, indent=2)

    print(f"\n✅ Mejor configuración guardada en: {best_config_file}")
    print(f"✅ Visualizaciones guardadas en: {plots_dir}")
    
    # 7. Generaciones de convergencia vs población
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='population_size', y='convergence_gen', data=df_results)
    plt.title('Generaciones de convergencia vs Tamaño de población')
    plt.xlabel('Tamaño de población')
    plt.ylabel('Generación de convergencia')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'convergence_vs_population.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Fitness vs generaciones por instancia
    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='convergence_gen', y='fitness', hue='p_bias', size='population_size',
                        data=instance_data, sizes=(50, 200), alpha=0.7)
        plt.title(f'Fitness vs Generaciones para {instance}')
        plt.xlabel('Generación de convergencia')
        plt.ylabel('Fitness (menor es mejor)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'{instance}_fitness_vs_generations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 9. Proyección de hiperparámetros (cubo ND) usando PCA
    # Se seleccionan todas las columnas de hiperparámetros
    hyperparams = [col for col in df_results.columns 
                   if col not in ['instance', 'fitness', 'convergence_gen', 'execution_time', 'num_routes', 'num_used_depots', 'is_valid', 'norm_fitness']]
    if hyperparams:
        pca = PCA(n_components=3)
        hyperparams_values = df_results[hyperparams].values
        pca_result = pca.fit_transform(hyperparams_values)
        df_results['pca_x'] = pca_result[:, 0]
        df_results['pca_y'] = pca_result[:, 1]
        df_results['pca_z'] = pca_result[:, 2]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_results['pca_x'], df_results['pca_y'], df_results['pca_z'], 
                             c=df_results['fitness'], cmap='viridis', s=50, alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Fitness (menor es mejor)')
        ax.set_title('Proyección de hiperparámetros (PCA 3D) - Cubo ND')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        plt.savefig(os.path.join(plots_dir, 'hyperparameter_cube_nd.png'), dpi=300, bbox_inches='tight')
        plt.close()
def generate_advanced_visualizations(df_results, results_dir):
    """
    Genera visualizaciones avanzadas para analizar el espacio de búsqueda de hiperparámetros.

    Args:
        df_results: DataFrame con resultados
        results_dir: Directorio donde guardar las visualizaciones
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    import pandas as pd
    from scipy.interpolate import Rbf
    from scipy.optimize import minimize
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d import Axes3D
    
    plots_dir = os.path.join(results_dir, 'advanced_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Lista de hiperparámetros
    params = ['population_size', 'elite_percent', 'mutants_percent', 'max_generations', 'p_bias']
    instances = df_results['instance'].unique()
    
    # 1. Modelo de superficie de respuesta con RBF (Radial Basis Function)
    param_pairs = [
        ('population_size', 'elite_percent'),
        ('elite_percent', 'mutants_percent'),
        ('population_size', 'p_bias')
    ]

    for instance in instances:
        instance_data = df_results[df_results['instance'] == instance]
        
        for param_x, param_y in param_pairs:
            try:
                plt.figure(figsize=(12, 10))
                
                # Obtener datos
                x = instance_data[param_x].values
                y = instance_data[param_y].values
                z = instance_data['fitness'].values
                
                # Evitar duplicados en las coordenadas x,y
                unique_coords = {}
                for i, (xi, yi) in enumerate(zip(x, y)):
                    if (xi, yi) not in unique_coords or z[i] < unique_coords[(xi, yi)]:
                        unique_coords[(xi, yi)] = z[i]
                
                x_unique = np.array([k[0] for k in unique_coords.keys()])
                y_unique = np.array([k[1] for k in unique_coords.keys()])
                z_unique = np.array(list(unique_coords.values()))
                
                if len(x_unique) > 5:  # Necesitamos un mínimo de puntos para la interpolación
                    # Crear malla para visualización
                    x_min, x_max = min(x_unique), max(x_unique)
                    y_min, y_max = min(y_unique), max(y_unique)
                    
                    # Expandir ligeramente los límites para mejor visualización
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    x_min -= x_range * 0.05
                    x_max += x_range * 0.05
                    y_min -= y_range * 0.05
                    y_max += y_range * 0.05
                    
                    # Crear malla
                    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                    
                    # Crear modelo RBF para interpolar la superficie
                    rbf = Rbf(x_unique, y_unique, z_unique, function='multiquadric', epsilon=2)
                    Z = rbf(X, Y)
                    
                    # Crear gráfico 3D de la superficie
                    ax = plt.axes(projection='3d')
                    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
                    
                    # Añadir puntos originales
                    ax.scatter(x_unique, y_unique, z_unique, c='red', s=50, depthshade=True)
                    
                    # Encontrar el mínimo de la superficie interpolada
                    def rbf_func(point):
                        return float(rbf(point[0], point[1]))
                    
                    x0 = [x_unique[np.argmin(z_unique)], y_unique[np.argmin(z_unique)]]
                    result = minimize(rbf_func, x0, bounds=[(x_min, x_max), (y_min, y_max)])
                    
                    if result.success:
                        min_x, min_y = result.x
                        min_z = rbf_func(result.x)
                        ax.scatter([min_x], [min_y], [min_z], c='gold', s=100, marker='*',
                                 label='Mínimo estimado')
                        
                        # Añadir anotación con los valores estimados óptimos
                        ax.text(min_x, min_y, min_z, 
                               f'({min_x:.2f}, {min_y:.2f})\nFitness: {min_z:.2f}',
                               color='black', fontsize=10, verticalalignment='bottom')
                    
                    ax.set_xlabel(param_x)
                    ax.set_ylabel(param_y)
                    ax.set_zlabel('Fitness')
                    ax.set_title(f'Superficie de Respuesta para {instance}: {param_x} vs {param_y}', fontsize=14)
                    
                    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Fitness')
                    
                    plt.savefig(os.path.join(plots_dir, f'{instance}_{param_x}_{param_y}_response_surface.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # También crear una vista de contorno 2D
                    plt.figure(figsize=(10, 8))
                    contour = plt.contourf(X, Y, Z, 20, cmap='viridis_r')
                    plt.colorbar(contour, label='Fitness (menor es mejor)')
                    
                    # Añadir puntos originales
                    plt.scatter(x_unique, y_unique, c=z_unique, cmap='viridis_r', 
                               s=50, edgecolor='k', alpha=0.7)
                    
                    # Marcar el mínimo estimado
                    if result.success:
                        plt.scatter(min_x, min_y, c='red', s=100, marker='*', 
                                   edgecolor='white', label='Óptimo estimado')
                        plt.annotate(f'({min_x:.2f}, {min_y:.2f})\nFitness: {min_z:.2f}',
                                   (min_x, min_y), xytext=(10, 10), textcoords="offset points",
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
                    
                    plt.title(f'Contorno de Superficie de Respuesta para {instance}: {param_x} vs {param_y}')
                    plt.xlabel(param_x)
                    plt.ylabel(param_y)
                    plt.legend()
                    
                    plt.savefig(os.path.join(plots_dir, f'{instance}_{param_x}_{param_y}_response_contour.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Error al generar superficie de respuesta para {param_x} vs {param_y}: {str(e)}")
    
    # 2. Visualización de la trayectoria en el espacio de hiperparámetros
    # Este análisis es útil si tienes información de la historia de búsqueda
    # En este caso, simularemos la trayectoria ordenando por fitness
    try:
        for instance in instances:
            instance_data = df_results[df_results['instance'] == instance].copy()
            
            # Ordenar por fitness (de peor a mejor) para simular la trayectoria de búsqueda
            trajectory_data = instance_data.sort_values('fitness', ascending=False).reset_index(drop=True)
            
            if len(trajectory_data) >= 5:  # Necesitamos un mínimo de puntos
                # Normalizar datos para visualización
                trajectory_norm = trajectory_data[params].copy()
                for param in params:
                    min_val = trajectory_data[param].min()
                    max_val = trajectory_data[param].max()
                    if max_val > min_val:
                        trajectory_norm[param] = (trajectory_data[param] - min_val) / (max_val - min_val)
                    else:
                        trajectory_norm[param] = 0.5
                
                # Visualizar la trayectoria en un espacio 3D usando los 3 primeros parámetros
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                x = trajectory_norm[params[0]].values
                y = trajectory_norm[params[1]].values
                z = trajectory_norm[params[2]].values
                
                # Crear un mapa de colores basado en el rango de fitness
                norm = plt.Normalize(trajectory_data['fitness'].min(), trajectory_data['fitness'].max())
                colors = plt.cm.viridis_r(norm(trajectory_data['fitness']))
                
                # Graficar los puntos
                ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
                
                # Conectar puntos con líneas para mostrar la trayectoria
                for i in range(len(x) - 1):
                    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 'k-', alpha=0.3)
                
                # Marcar el punto inicial y final
                ax.scatter(x[0], y[0], z[0], c='red', s=100, label='Inicio')
                ax.scatter(x[-1], y[-1], z[-1], c='green', s=100, label='Mejor')
                
                ax.set_xlabel(f'{params[0]} (normalizado)')
                ax.set_ylabel(f'{params[1]} (normalizado)')
                ax.set_zlabel(f'{params[2]} (normalizado)')
                ax.set_title(f'Trayectoria de búsqueda para {instance}', fontsize=14)
                ax.legend()
                
                plt.savefig(os.path.join(plots_dir, f'{instance}_search_trajectory.png'), dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print(f"Error al generar visualización de trayectoria para {instance}: {str(e)}")
                
    # 3. Proyección de hiperparámetros con UMAP/t-SNE para mejor visualización
    # Crear visualización de comparación de rendimiento entre instancias
    try:
        if len(instances) > 1:
            # Para cada instancia, encontrar la mejor configuración
            best_configs = []
            for instance in instances:
                instance_data = df_results[df_results['instance'] == instance]
                best_idx = instance_data['fitness'].idxmin()
                best_configs.append(instance_data.loc[best_idx])
            
            best_df = pd.DataFrame(best_configs)
            
            # Gráfico de radar para comparar la mejor configuración entre instancias
            if len(best_df) > 1:
                # Preparar datos para el gráfico de radar
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Normalizar valores de hiperparámetros para el radar
                normalized_df = best_df[params].copy()
                for param in params:
                    min_val = df_results[param].min()
                    max_val = df_results[param].max()
                    if max_val > min_val:
                        normalized_df[param] = (best_df[param] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[param] = best_df[param] * 0 + 0.5  # Valores constantes se mapean a 0.5
                
                # Configurar ejes del radar
                angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
                ax.set_thetagrids(np.degrees(angles), params)
                
                # Añadir líneas concéntricas
                ax.set_rlabel_position(0)
                ax.set_rticks([0.25, 0.5, 0.75])
                ax.set_rlim(0, 1)
                
                # Dibujar líneas para cada instancia
                for i, instance in enumerate(best_df['instance']):
                    values = normalized_df.iloc[i].values.tolist()
                    values += values[:1]  # Cerrar el polígono
                    angles_closed = angles + angles[:1]
                    ax.plot(angles_closed, values, 'o-', linewidth=2, label=instance)
                    ax.fill(angles_closed, values, alpha=0.1)
                
                plt.title('Comparación de Mejores Configuraciones por Instancia', fontsize=14, y=1.08)
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                plt.savefig(os.path.join(plots_dir, 'best_configs_radar.png'), dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print(f"Error al generar visualización de radar para mejores configuraciones: {str(e)}")
    
    # 4. Visualización de la evolución del fitness
    try:
        if 'iteration' in df_results.columns:  # Si tienes datos de iteraciones
            plt.figure(figsize=(12, 8))
            
            # Filtrar algunos de los mejores experimentos para cada instancia
            plot_data = []
            for instance in instances:
                instance_data = df_results[df_results['instance'] == instance]
                # Tomar las 3 mejores configuraciones
                best_configs = instance_data.nsmallest(3, 'fitness')
                plot_data.append(best_configs)
            
            plot_df = pd.concat(plot_data)
            
            # Graficar la evolución del fitness por iteración
            sns.lineplot(data=plot_df, x='iteration', y='fitness', hue='instance', style='config_id', 
                       markers=True, dashes=False)
            
            plt.title('Evolución del Fitness para las Mejores Configuraciones')
            plt.xlabel('Iteración')
            plt.ylabel('Fitness (menor es mejor)')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, 'fitness_evolution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error al generar visualización de evolución del fitness: {str(e)}")
    
    # 5. Caracterización del espacio de soluciones
    try:
        for instance in instances:
            instance_data = df_results[df_results['instance'] == instance]
            
            if len(instance_data) > 10:  # Necesitamos suficientes puntos
                # Crear un gráfico de dispersión con densidad
                plt.figure(figsize=(12, 10))
                
                # Crear un KDE para visualizar la densidad de soluciones buenas
                x = instance_data['fitness'].values
                
                # Eliminar valores atípicos para mejor visualización
                q1, q3 = np.percentile(x, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_x = x[(x >= lower_bound) & (x <= upper_bound)]
                
                if len(filtered_x) > 5:
                    sns.histplot(filtered_x, kde=True, color='teal')
                    plt.axvline(x.min(), color='red', linestyle='--', label='Mejor fitness')
                    
                    # Añadir estadísticas
                    plt.annotate(f'Mejor: {x.min():.2f}\nMedia: {x.mean():.2f}\nMediana: {np.median(x):.2f}\nDesv.Est: {x.std():.2f}',
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    
                    plt.title(f'Distribución del Fitness para {instance}')
                    plt.xlabel('Fitness')
                    plt.ylabel('Frecuencia')
                    plt.legend()
                    
                    plt.savefig(os.path.join(plots_dir, f'{instance}_fitness_distribution.png'), dpi=300, bbox_inches='tight')
                    plt.close()
    except Exception as e:
        print(f"Error al generar visualización de distribución del fitness: {str(e)}")
    
    # 6. Análisis de sensibilidad para cada hiperparámetro
    try:
        # Cuantificar cuánto cambia el fitness al variar cada hiperparámetro
        sensitivity_data = {}
        
        for param in params:
            # Obtener valores únicos del hiperparámetro
            unique_values = sorted(df_results[param].unique())
            
            if len(unique_values) > 1:
                # Para cada valor único, calcular estadísticas de fitness
                values = []
                means = []
                stds = []
                mins = []
                
                for val in unique_values:
                    subset = df_results[df_results[param] == val]
                    values.append(val)
                    means.append(subset['fitness'].mean())
                    stds.append(subset['fitness'].std())
                    mins.append(subset['fitness'].min())
                
                # Calcular la variación relativa de fitness entre el mejor y peor valor
                best_idx = np.argmin(means)
                worst_idx = np.argmax(means)
                if means[worst_idx] > 0:  # Evitar división por cero
                    relative_variation = (means[worst_idx] - means[best_idx]) / means[worst_idx] * 100
                else:
                    relative_variation = 0
                
                sensitivity_data[param] = {
                    'values': values,
                    'means': means,
                    'stds': stds,
                    'mins': mins,
                    'best_value': values[best_idx],
                    'relative_variation': relative_variation
                }
        
        # Crear gráfico de sensibilidad
        if sensitivity_data:
            # Gráfico de barras para mostrar la variación relativa
            variations = [data['relative_variation'] for data in sensitivity_data.values()]
            param_names = list(sensitivity_data.keys())
            
            sorted_indices = np.argsort(variations)[::-1]  # Ordenar de mayor a menor sensibilidad
            sorted_params = [param_names[i] for i in sorted_indices]
            sorted_variations = [variations[i] for i in sorted_indices]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(sorted_params, sorted_variations, color='teal')
            
            # Añadir etiquetas con los mejores valores
            for i, bar in enumerate(bars):
                param = sorted_params[i]
                best_val = sensitivity_data[param]['best_value']
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'Mejor: {best_val}', ha='center', va='bottom', rotation=0)
            
            plt.title('Análisis de Sensibilidad de Hiperparámetros')
            plt.xlabel('Hiperparámetro')
            plt.ylabel('Variación Relativa (%)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Gráficos detallados de sensibilidad para cada parámetro
            for param, data in sensitivity_data.items():
                plt.figure(figsize=(10, 6))
                
                # Graficar el fitness medio con barras de error
                plt.errorbar(data['values'], data['means'], yerr=data['stds'], 
                           fmt='-o', color='teal', capsize=5, label='Fitness medio ± std')
                
                # Graficar el mejor fitness para cada valor
                plt.plot(data['values'], data['mins'], 'r--o', label='Mejor fitness')
                
                # Marcar el mejor valor
                best_idx = np.argmin(data['means'])
                plt.axvline(x=data['values'][best_idx], color='green', linestyle='--', 
                          label=f'Mejor valor: {data["values"][best_idx]}')
                
                plt.title(f'Análisis de Sensibilidad para {param}')
                plt.xlabel(param)
                plt.ylabel('Fitness (menor es mejor)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(plots_dir, f'sensitivity_{param}.png'), dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print(f"Error al generar análisis de sensibilidad: {str(e)}")
    
    print(f"✅ Visualizaciones avanzadas guardadas en: {plots_dir}")


# Para integrar con tu script existente:
# generate_advanced_visualizations(df_results, output_dir)

def generate_summary_report(df_results, results_dir):
    """
    Genera un informe resumido con las conclusiones clave del análisis de hiperparámetros.
    
    Args:
        df_results: DataFrame con resultados
        results_dir: Directorio donde guardar el informe
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Crear directorio para el informe
    report_dir = os.path.join(results_dir, 'summary_report')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Lista de hiperparámetros e instancias
    params = ['population_size', 'elite_percent', 'mutants_percent', 'max_generations', 'p_bias']
    instances = df_results['instance'].unique()
    
    # 1. Crear archivo de texto con el informe
    report_file = os.path.join(report_dir, 'hyperparameter_analysis_summary.md')
    
    with open(report_file, 'w') as f:
        # Encabezado
        f.write("# Análisis de Hiperparámetros: Informe Resumido\n\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Información general
        f.write("## Información General\n\n")
        f.write(f"- Total de configuraciones evaluadas: {len(df_results)}\n")
        f.write(f"- Instancias analizadas: {len(instances)}\n")
        f.write(f"- Hiperparámetros optimizados: {len(params)}\n\n")
        
        # Valores óptimos por instancia
        f.write("## Mejores Configuraciones por Instancia\n\n")
        
        # Crear tabla para la mejor configuración de cada instancia
        f.write("| Instancia | Fitness | ")
        for param in params:
            f.write(f"{param} | ")
        f.write("Tiempo (s) | Generación Convergencia |\n")
        
        f.write("|" + "---|" * (len(params) + 4) + "\n")
        
        for instance in instances:
            instance_data = df_results[df_results['instance'] == instance]
            best_config = instance_data.loc[instance_data['fitness'].idxmin()]
            
            f.write(f"| {instance} | {best_config['fitness']:.2f} | ")
            for param in params:
                f.write(f"{best_config[param]} | ")
            f.write(f"{best_config['execution_time']:.2f} | {best_config['convergence_gen']} |\n")
        
        f.write("\n")
        
        # Mejor configuración global
        f.write("## Mejor Configuración Global\n\n")
        
        # Normalizar fitness para cada instancia y encontrar la mejor configuración general
        best_configs = pd.DataFrame()
        for instance in instances:
            instance_data = df_results[df_results['instance'] == instance]
            min_fitness = instance_data['fitness'].min()
            max_fitness = instance_data['fitness'].max()
            if max_fitness > min_fitness:
                instance_data['norm_fitness'] = (instance_data['fitness'] - min_fitness) / (max_fitness - min_fitness)
            else:
                instance_data['norm_fitness'] = 0
            best_configs = pd.concat([best_configs, instance_data])
        
        avg_perf = best_configs.groupby(params)['norm_fitness'].mean().reset_index()
        best_overall = avg_perf.loc[avg_perf['norm_fitness'].idxmin()]
        
        f.write("La mejor configuración global (promediando rendimiento normalizado en todas las instancias):\n\n")
        f.write("```\n")
        for param in params:
            f.write(f"{param}: {best_overall[param]}\n")
        f.write(f"Rendimiento normalizado: {best_overall['norm_fitness']:.4f}\n")
        f.write("```\n\n")
        
        # Análisis de sensibilidad
        f.write("## Análisis de Sensibilidad\n\n")
        f.write("Impacto relativo de cada hiperparámetro en el rendimiento (mayor porcentaje = más influencia):\n\n")
        
        # Calcular sensibilidad
        sensitivity_data = {}
        for param in params:
            unique_values = sorted(df_results[param].unique())
            if len(unique_values) > 1:
                means = []
                for val in unique_values:
                    subset = df_results[df_results[param] == val]
                    means.append(subset['fitness'].mean())
                
                best_idx = np.argmin(means)
                worst_idx = np.argmax(means)
                if means[worst_idx] > 0:
                    relative_variation = (means[worst_idx] - means[best_idx]) / means[worst_idx] * 100
                else:
                    relative_variation = 0
                
                sensitivity_data[param] = {
                    'best_value': unique_values[best_idx],
                    'relative_variation': relative_variation
                }
        
        if sensitivity_data:
            # Ordenar por sensibilidad
            sorted_params = sorted(sensitivity_data.keys(), 
                                 key=lambda x: sensitivity_data[x]['relative_variation'], 
                                 reverse=True)
            
            f.write("| Hiperparámetro | Impacto (%) | Mejor Valor |\n")
            f.write("|---|---|---|\n")
            
            for param in sorted_params:
                data = sensitivity_data[param]
                f.write(f"| {param} | {data['relative_variation']:.2f}% | {data['best_value']} |\n")
            
            f.write("\n")
            
            # Recomendar valores
            f.write("### Recomendaciones de Valores\n\n")
            
            for param in sorted_params:
                data = sensitivity_data[param]
                if data['relative_variation'] > 10:  # Parámetros con alto impacto
                    f.write(f"- **{param}**: Configurar en {data['best_value']} (impacto alto: {data['relative_variation']:.1f}%)\n")
                elif data['relative_variation'] > 5:  # Parámetros con impacto medio
                    f.write(f"- **{param}**: Preferiblemente usar {data['best_value']} (impacto medio: {data['relative_variation']:.1f}%)\n")
                else:  # Parámetros con bajo impacto
                    f.write(f"- {param}: Usar {data['best_value']} o cualquier valor cercano (impacto bajo: {data['relative_variation']:.1f}%)\n")
            
            f.write("\n")
        
        # Análisis de tiempo de ejecución
        f.write("## Análisis de Tiempo de Ejecución\n\n")
        
        # Relación entre tamaño de población y tiempo
        pop_vs_time = df_results.groupby('population_size')['execution_time'].mean().reset_index()
        
        if not pop_vs_time.empty:
            min_time_config = pop_vs_time.loc[pop_vs_time['execution_time'].idxmin()]
            max_time_config = pop_vs_time.loc[pop_vs_time['execution_time'].idxmax()]
            
            speedup = max_time_config['execution_time'] / min_time_config['execution_time'] if min_time_config['execution_time'] > 0 else 1
            
            f.write(f"- Configuración más rápida: población = {min_time_config['population_size']}, "
                  f"tiempo promedio = {min_time_config['execution_time']:.2f}s\n")
            f.write(f"- Configuración más lenta: población = {max_time_config['population_size']}, "
                  f"tiempo promedio = {max_time_config['execution_time']:.2f}s\n")
            f.write(f"- Aceleración potencial: {speedup:.2f}x\n\n")
        
        # Análisis de convergencia
        f.write("## Análisis de Convergencia\n\n")
        
        if 'convergence_gen' in df_results.columns:
            avg_convergence = df_results.groupby('population_size')['convergence_gen'].mean().reset_index()
            
            if not avg_convergence.empty:
                early_conv = avg_convergence.loc[avg_convergence['convergence_gen'].idxmin()]
                late_conv = avg_convergence.loc[avg_convergence['convergence_gen'].idxmax()]
                
                f.write(f"- Convergencia más rápida: población = {early_conv['population_size']}, "
                      f"generación promedio = {early_conv['convergence_gen']:.1f}\n")
                f.write(f"- Convergencia más lenta: población = {late_conv['population_size']}, "
                      f"generación promedio = {late_conv['convergence_gen']:.1f}\n\n")
        
        # Conclusiones
        f.write("## Conclusiones y Recomendaciones\n\n")
        
        # Ordenar parámetros por importancia
        if sensitivity_data:
            most_important = sorted_params[0] if sorted_params else "ninguno"
            least_important = sorted_params[-1] if sorted_params else "ninguno"
            
            f.write(f"1. El hiperparámetro con mayor impacto en el rendimiento es **{most_important}**.\n")
            f.write(f"2. El hiperparámetro con menor impacto es **{least_important}**.\n\n")
        
        # Recomendación de configuración
        f.write("### Configuración Recomendada\n\n")
        f.write("```\n")
        for param in params:
            f.write(f"{param}: {best_overall[param]}\n")
        f.write("```\n\n")
        
        # Tiempo vs calidad
        f.write("### Equilibrio entre Tiempo y Calidad\n\n")
        
        # Intentar encontrar una configuración con buen equilibrio entre tiempo y calidad
        if 'norm_fitness' in best_configs.columns and 'execution_time' in best_configs.columns:
            # Normalizar tiempo de ejecución
            min_time = best_configs['execution_time'].min()
            max_time = best_configs['execution_time'].max()
            
            if max_time > min_time:
                best_configs['norm_time'] = (best_configs['execution_time'] - min_time) / (max_time - min_time)
            else:
                best_configs['norm_time'] = 0
            
            # Calcular puntuación combinada (50% fitness, 50% tiempo)
            best_configs['combined_score'] = 0.5 * best_configs['norm_fitness'] + 0.5 * best_configs['norm_time']
            
            # Encontrar la mejor configuración equilibrada
            balanced_idx = best_configs['combined_score'].idxmin()
            balanced_config = best_configs.loc[balanced_idx]
            
            f.write("Para un equilibrio entre calidad de resultados y tiempo de ejecución:\n\n")
            f.write("```\n")
            for param in params:
                f.write(f"{param}: {balanced_config[param]}\n")
            f.write(f"Tiempo estimado: {balanced_config['execution_time']:.2f}s\n")
            f.write(f"Calidad relativa: {(1-balanced_config['norm_fitness'])*100:.1f}%\n")
            f.write("```\n\n")
        
        # Recomendaciones específicas por instancia
        f.write("### Recomendaciones por Tipo de Problema\n\n")
        
        f.write("Si necesitas optimizar para:\n\n")
        f.write("- **Problemas grandes**: Prioriza configuraciones con mayor tamaño de población\n")
        f.write("- **Tiempo limitado**: Utiliza tamaños de población más pequeños con porcentajes de élites más altos\n")
        f.write("- **Máxima calidad**: Utiliza la configuración global recomendada\n\n")
        
        # Mejoras futuras
        f.write("## Sugerencias para Futuras Optimizaciones\n\n")
        
        f.write("1. **Exploración adaptativa**: Considerar implementar mecanismos adaptativos para los parámetros más sensibles\n")
        f.write(f"2. **Enfoque en {most_important if sensitivity_data else 'parámetros clave'}**: Realizar una búsqueda más fina para este parámetro\n")
        f.write("3. **Hibridización**: Considerar combinar el algoritmo genético con búsquedas locales\n")
        f.write("4. **Paralelización**: Evaluar la implementación de evaluación paralela para poblaciones grandes\n")
    
    print(f"✅ Informe de análisis guardado en: {report_file}")
    
    # 2. Crear visualización resumen para las conclusiones principales
    try:
        # Crear un dashboard visual con las conclusiones clave
        plt.figure(figsize=(12, 10))
        
        # Título principal
        plt.suptitle('Resumen de Análisis de Hiperparámetros', fontsize=16, y=0.98)
        
        # 1. Gráfica superior izquierda: Importancia de hiperparámetros
        if sensitivity_data:
            plt.subplot(2, 2, 1)
            sorted_params = sorted(sensitivity_data.keys(), 
                                 key=lambda x: sensitivity_data[x]['relative_variation'], 
                                 reverse=True)
            sorted_variations = [sensitivity_data[param]['relative_variation'] for param in sorted_params]
            
            bars = plt.bar(sorted_params, sorted_variations, color='skyblue')
            plt.title('Importancia de Hiperparámetros', fontsize=12)
            plt.ylabel('Impacto Relativo (%)')
            plt.xticks(rotation=45)
            
            # Añadir etiquetas
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 2. Gráfica superior derecha: Mejor configuración
        plt.subplot(2, 2, 2)
        best_values = [best_overall[param] for param in params]
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        
        wedges, texts, autotexts = plt.pie(
            [1] * len(params), 
            labels=params,
            autopct=lambda pct: f"{pct:.1f}%",
            colors=colors,
            wedgeprops=dict(width=0.5)
        )
        
        # Ajustar etiquetas
        for i, autotext in enumerate(autotexts):
            autotext.set_text(f"{best_values[i]}")
        
        plt.title('Mejor Configuración Global', fontsize=12)
        
        # 3. Gráfica inferior izquierda: Tiempo vs. Población
        plt.subplot(2, 2, 3)
        if 'population_size' in df_results.columns and 'execution_time' in df_results.columns:
            pop_sizes = sorted(df_results['population_size'].unique())
            exec_times = []
            
            for pop in pop_sizes:
                exec_times.append(df_results[df_results['population_size'] == pop]['execution_time'].mean())
            
            plt.plot(pop_sizes, exec_times, 'o-', color='teal')
            plt.title('Tiempo vs. Tamaño de Población', fontsize=12)
            plt.xlabel('Tamaño de Población')
            plt.ylabel('Tiempo Promedio (s)')
            plt.grid(True, alpha=0.3)
        
        # 4. Gráfica inferior derecha: Mejor fitness por instancia
        plt.subplot(2, 2, 4)
        instance_best_fitness = []
        
        for instance in instances:
            instance_data = df_results[df_results['instance'] == instance]
            instance_best_fitness.append({
                'instance': instance,
                'best_fitness': instance_data['fitness'].min()
            })
        
        best_fitness_df = pd.DataFrame(instance_best_fitness)
        
        if not best_fitness_df.empty:
            best_fitness_df = best_fitness_df.sort_values('best_fitness')
            
            bars = plt.bar(best_fitness_df['instance'], best_fitness_df['best_fitness'], color='lightgreen')
            plt.title('Mejor Fitness por Instancia', fontsize=12)
            plt.ylabel('Fitness (menor es mejor)')
            plt.xticks(rotation=45)
            
            # Añadir etiquetas
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
        plt.savefig(os.path.join(report_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard resumen guardado en: {os.path.join(report_dir, 'summary_dashboard.png')}")
    except Exception as e:
        print(f"Error al generar dashboard resumen: {str(e)}")
    
    # 3. Crear visualización interactiva HTML (opcional)
    try:
        html_report = os.path.join(report_dir, 'interactive_summary.html')
        
        with open(html_report, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Análisis de Hiperparámetros - Resumen Interactivo</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 20px; 
                        background-color: #f5f5f5;
                    }
                    .container { 
                        max-width: 1200px; 
                        margin: 0 auto; 
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }
                    h1 { color: #2c3e50; text-align: center; }
                    h2 { color: #3498db; margin-top: 30px; }
                    table { 
                        width: 100%; 
                        border-collapse: collapse; 
                        margin: 20px 0;
                    }
                    th, td { 
                        padding: 10px; 
                        border: 1px solid #ddd; 
                        text-align: left;
                    }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .highlight { background-color: #e8f4f8; font-weight: bold; }
                    .card {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px 0;
                        background-color: #fff;
                    }
                    .flex-container {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }
                    .flex-item {
                        flex: 0 0 48%;
                        margin-bottom: 20px;
                    }
                    .chart-container {
                        width: 100%;
                        height: 300px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        overflow: hidden;
                    }
                    @media (max-width: 768px) {
                        .flex-item {
                            flex: 0 0 100%;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Análisis de Hiperparámetros - Resumen Interactivo</h1>
                    
                    <div class="card">
                        <h2>Mejor Configuración Global</h2>
                        <div class="flex-container">
            ''')
            
            # Añadir mejor configuración global
            f.write('<div class="flex-item"><table>')
            f.write('<tr><th>Hiperparámetro</th><th>Valor Óptimo</th></tr>')
            
            for param in params:
                f.write(f'<tr class="highlight"><td>{param}</td><td>{best_overall[param]}</td></tr>')
            
            f.write('</table></div>')
            
            # Añadir espacio para gráfico (se cargará una imagen)
            f.write('''
                <div class="flex-item">
                    <div class="chart-container">
                        <img src="summary_dashboard.png" alt="Resumen visual" style="width:100%; height:auto;">
                    </div>
                </div>
            </div>
            </div>
            ''')
            
            # Añadir mejores configuraciones por instancia
            f.write('''
                <div class="card">
                    <h2>Mejores Configuraciones por Instancia</h2>
                    <table>
                        <tr>
                            <th>Instancia</th>
                            <th>Fitness</th>
            ''')
            
            for param in params:
                f.write(f'<th>{param}</th>')
            
            f.write('<th>Tiempo (s)</th></tr>')
            
            for instance in instances:
                instance_data = df_results[df_results['instance'] == instance]
                best_config = instance_data.loc[instance_data['fitness'].idxmin()]
                
                f.write(f'<tr><td>{instance}</td><td>{best_config["fitness"]:.2f}</td>')
                
                for param in params:
                    f.write(f'<td>{best_config[param]}</td>')
                
                f.write(f'<td>{best_config["execution_time"]:.2f}</td></tr>')
            
            f.write('</table></div>')
            
            # Sensibilidad de hiperparámetros
            if sensitivity_data:
                f.write('''
                    <div class="card">
                        <h2>Análisis de Sensibilidad</h2>
                        <table>
                            <tr>
                                <th>Hiperparámetro</th>
                                <th>Impacto (%)</th>
                                <th>Mejor Valor</th>
                                <th>Recomendación</th>
                            </tr>
                ''')
                
                for param in sorted_params:
                    data = sensitivity_data[param]
                    impact = data['relative_variation']
                    
                    # Determinar recomendación
                    if impact > 10:
                        recommendation = f"<strong>Crítico - Usar exactamente {data['best_value']}</strong>"
                    elif impact > 5:
                        recommendation = f"Importante - Preferir {data['best_value']}"
                    else:
                        recommendation = f"Flexible - {data['best_value']} o similar"
                    
                    # Estilo según impacto
                    if impact > 10:
                        row_class = "highlight"
                    else:
                        row_class = ""
                    
                    f.write(f'<tr class="{row_class}"><td>{param}</td><td>{impact:.2f}%</td>' + 
                          f'<td>{data["best_value"]}</td><td>{recommendation}</td></tr>')
                
                f.write('</table></div>')
            
            # Añadir sección de conclusiones
            f.write('''
                <div class="card">
                    <h2>Conclusiones y Recomendaciones</h2>
                    <div class="flex-container">
                        <div class="flex-item">
                            <h3>Recomendación General</h3>
                            <p>Para obtener los mejores resultados en todas las instancias, se recomienda la siguiente configuración:</p>
                            <table>
            ''')
            
            for param in params:
                f.write(f'<tr><td><strong>{param}</strong></td><td>{best_overall[param]}</td></tr>')
            
            f.write('''
                            </table>
                        </div>
                        <div class="flex-item">
                            <h3>Optimización por Objetivo</h3>
                            <table>
                                <tr><th>Objetivo</th><th>Recomendación</th></tr>
                                <tr><td>Máxima calidad</td><td>Usar la configuración global recomendada</td></tr>
                                <tr><td>Tiempo limitado</td><td>Reducir tamaño de población, aumentar élites</td></tr>
                                <tr><td>Problemas grandes</td><td>Aumentar tamaño de población</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
            ''')
            
            # Añadir sección de mejoras futuras
            f.write('''
                <div class="card">
                    <h2>Mejoras Futuras</h2>
                    <ul>
                        <li><strong>Exploración adaptativa</strong>: Implementar mecanismos adaptativos para los parámetros más sensibles</li>
                        <li><strong>Enfoque en parámetros clave</strong>: Realizar una búsqueda más fina para los parámetros de mayor impacto</li>
                        <li><strong>Hibridización</strong>: Combinar el algoritmo genético con búsquedas locales</li>
                        <li><strong>Paralelización</strong>: Implementar evaluación paralela para poblaciones grandes</li>
                    </ul>
                </div>
            ''')
            
            # Cerrar documento HTML
            f.write('''
                </div>
            </body>
            </html>
            ''')
        
        print(f"✅ Informe interactivo guardado en: {html_report}")
    except Exception as e:
        print(f"Error al generar informe interactivo: {str(e)}")
    
    print(f"\n✅ Proceso de generación de informe completado")

def main():
    # Directorio donde se encuentran los archivos de resultados
    results_dir = './hyperparameter_results'
    output_dir = './hyperparameter_results'
    
    if not os.path.exists(results_dir):
        print(f"❌ No se encontró el directorio de resultados: {results_dir}")
        sys.exit(1)
    
    df_results = load_results_from_files(results_dir)
    
    if df_results.empty:
        print("❌ No se pudieron cargar resultados.")
        sys.exit(1)
    
    csv_path = os.path.join(output_dir, 'hyperparameter_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en: {csv_path}")
    
    print("Generando visualizaciones básicas...")
    generate_visualizations(df_results, output_dir)
    
    print("Generando visualizaciones mejoradas...")
    generate_advanced_visualizations(df_results, output_dir)
    
    print("Generando visualizaciones avanzadas...")
    generate_advanced_visualizations(df_results, output_dir)
    
    # Crear informe de resultados
    print("Generando informe de resultados...")
    #generate_summary_report(df_results, output_dir)
    
    print("\n✅ Proceso completado")
        
    
    
if __name__ == "__main__":
    main()