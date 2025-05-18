import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import os
import json
import time
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm

# ----------------- FUNCIONES PARA LA EJECUCION DEL PROGRAMA -----------------

def run_all_instances(data_dir, instance_files, population_size, elite_percent, mutants_percent, max_generations, results, use_refinement=True):
    """Ejecuta el algoritmo BRKGA en todas las instancias, continuando desde donde se quedó anteriormente"""
    
    # Verificar si hay resultados previos
    resultados_file = '../resultados.json'
    completed_instances = set()
    
    if os.path.exists(resultados_file):
        try:
            with open(resultados_file, 'r') as f:
                previous_results = json.load(f)
                
            # Actualizar el diccionario de resultados con los resultados previos
            for instance, data in previous_results.items():
                if 'error' not in data:
                    print(f"✅ Ya procesado: {instance} - Fitness: {data['fitness']}")
                    results[instance] = data
                    completed_instances.add(instance)
                else:
                    print(f"⚠️ Error previo: {instance} - {data['error']}")
                    
            # Guardar número de instancias ya procesadas
            print(f"\nSe encontraron {len(completed_instances)} instancias ya procesadas.")
        except Exception as e:
            print(f"Error al cargar resultados previos: {str(e)}")
    
    # Filtrar instancias que faltan por procesar
    pending_instances = [f for f in instance_files if f not in completed_instances]
    
    if not pending_instances:
        print("\n✅ Todas las instancias ya han sido procesadas!")
        return results
    
    print(f"Faltan {len(pending_instances)} instancias por procesar.")
    
    # Ejecutar BRKGA para cada instancia pendiente
    for instance_file in tqdm(pending_instances, desc="Procesando instancias pendientes"):
        instance_name = instance_file
        file_path = os.path.join(data_dir, instance_file)
        
        print(f"\n{'='*80}")
        print(f"Procesando instancia: {instance_name}")
        print(f"{'='*80}")
        
        try:
            # Cargar datos de la instancia
            mdvrp_data = parse_mdvrp_file(file_path)
            print(f"Instancia cargada: {mdvrp_data['num_customers']} clientes, {mdvrp_data['num_depots']} depósitos")
            
            # Calcular early stopping adaptativo
            early_stop = calculate_early_stopping(mdvrp_data['num_customers'], mdvrp_data['num_depots'])
            print(f"Early stopping adaptativo: {early_stop} generaciones sin mejora")
            
            # Resolver con BRKGA
            start_time = time.time()
            brkga = BRKGA_MDVRP(
                mdvrp_data, 
                population_size=population_size, 
                elite_percent=elite_percent, 
                mutants_percent=mutants_percent,
                use_refinement=use_refinement
            )
            
            # Ejecutar el algoritmo con early stopping adaptativo
            best_solution, best_distance, best_chromosome, fitness_history, convergence_gen = brkga.solve(
                generations=max_generations, 
                verbose=True,
                early_stopping=early_stop
            )
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            
            # Guardar resultados
            results[instance_name] = {
                'instance': instance_name,
                'fitness': best_distance,
                'generation': convergence_gen,
                'population_size': population_size,
                'execution_time': execution_time,
                'convergence_history': fitness_history,
                'best_chromosome': best_chromosome.tolist(),
                'best_solution': [
                    {
                        'depot_id': route['depot_id'],
                        'customers': route['customers'],
                        'load': route['load'],
                        'distance': route['distance']
                    } for route in best_solution
                ],
                'use_refinement': use_refinement,
                'early_stopping': early_stop
            }
            
            print(f"✅ Completado: {instance_name} - Fitness: {best_distance:.2f}, Tiempo: {execution_time:.2f}s")
            
            # Guardar los resultados parciales después de cada instancia
            with open(resultados_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            import traceback
            print(f"❌ Error procesando {instance_name}: {str(e)}")
            print(traceback.format_exc())  # Imprimir el traceback completo
            results[instance_name] = {
                'instance': instance_name,
                'error': str(e)
            }
            
            # También guardar después de errores
            with open(resultados_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    return results

def run_single_instance(data_dir, instance_file, population_size, elite_percent, mutants_percent, max_generations, use_refinement=True):
    """Ejecuta el algoritmo BRKGA en una sola instancia y visualiza los resultados"""
    file_path = os.path.join(data_dir, instance_file)
    
    print(f"\n{'='*80}")
    print(f"Procesando instancia: {instance_file}")
    print(f"{'='*80}")
    
    try:
        # Cargar datos de la instancia
        mdvrp_data = parse_mdvrp_file(file_path)
        print(f"Instancia cargada: {mdvrp_data['num_customers']} clientes, {mdvrp_data['num_depots']} depósitos")
        
        # Visualizar instancia original
        print("Visualizando instancia original...")
        plot_mdvrp_instance(mdvrp_data, show_demand=True)
        
        # Calcular early stopping adaptativo
        early_stop = calculate_early_stopping(mdvrp_data['num_customers'], mdvrp_data['num_depots'])
        
        # Resolver con BRKGA
        print(f"\nIniciando algoritmo BRKGA (refinamiento: {'activado' if use_refinement else 'desactivado'})...")
        print(f"Early stopping adaptativo: {early_stop} generaciones sin mejora")
        
        start_time = time.time()
        brkga = BRKGA_MDVRP(
            mdvrp_data, 
            population_size=population_size, 
            elite_percent=elite_percent, 
            mutants_percent=mutants_percent,
            use_refinement=use_refinement
        )
        
        # Ejecutar el algoritmo con early stopping adaptativo
        best_solution, best_distance, best_chromosome, fitness_history, convergence_gen = brkga.solve(
            generations=max_generations, 
            verbose=True,
            early_stopping=early_stop
        )
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Visualizar la convergencia
        plt.figure(figsize=(10, 5))
        plt.plot(fitness_history)
        plt.title(f'Convergencia del BRKGA - {instance_file}')
        plt.xlabel('Generación')
        plt.ylabel('Distancia total')
        plt.axvline(x=convergence_gen, color='r', linestyle='--', 
                   label=f'Mejor solución (Gen {convergence_gen})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'convergencia_{instance_file.replace(".dat", "")}.png', dpi=300)
        plt.show()
        
        # Guardar resultados
        result = {
            'instance': instance_file,
            'fitness': best_distance,
            'generation': convergence_gen,
            'population_size': population_size,
            'execution_time': execution_time,
            'convergence_history': fitness_history,
            'best_chromosome': best_chromosome.tolist(),
            'best_solution': best_solution,
            'data': mdvrp_data,
            'use_refinement': use_refinement,
            'early_stopping': early_stop
        }
        
        print(f"\n✅ Completado: {instance_file}")
        print(f"   Fitness: {best_distance:.2f}")
        print(f"   Tiempo: {execution_time:.2f}s")
        print(f"   Convergencia en generación: {convergence_gen}")
        print(f"   Early stopping: {early_stop}")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ Error procesando {instance_file}: {str(e)}")
        print(traceback.format_exc())  # Imprimir el traceback completo
        return {'instance': instance_file, 'error': str(e)}

def visualize_instance_solution(result, data_dir):
    """Visualiza la solución para una instancia específica"""
    if 'error' in result:
        print(f"❌ No se puede visualizar debido a un error: {result['error']}")
        return
    
    mdvrp_data = result['data']
    solution = result['best_solution']
    
    # Visualizar la solución
    print("\nVisualizando mejor solución encontrada...")
    visualize_routes(mdvrp_data, solution, show_demand=True, show_route_info=True)
    
    # Validar la solución
    print("\nValidando la solución...")
    is_valid, violations = debug_solution(mdvrp_data, solution, verbose=True)
    
    # Guardar la solución en un archivo
    solution_file = os.path.join(data_dir, f"{result['instance'].replace('.dat', '')}_solution.json")
    
    # Convertir a formato serializable
    solution_to_save = {
        'instance': result['instance'],
        'fitness': result['fitness'],
        'execution_time': result['execution_time'],
        'routes': [
            {
                'depot_id': route['depot_id'],
                'customers': route['customers'],
                'load': route['load'],
                'distance': route['distance']
            } for route in solution
        ],
        'is_valid': is_valid,
        'violations': violations
    }
    
    with open(solution_file, 'w') as f:
        json.dump(solution_to_save, f, indent=2)
    
    print(f"\n✅ Solución guardada en: {solution_file}")
    
    # Visualizar estadísticas adicionales
    print("\nGenerando visualizaciones adicionales...")
    
    # 1. Distribución de cargas por ruta
    plt.figure(figsize=(10, 6))
    loads = [route['load'] for route in solution]
    max_load = mdvrp_data['vehicle_info'][0]['max_load']
    
    plt.bar(range(1, len(loads)+1), loads, color='skyblue')
    plt.axhline(y=max_load, color='r', linestyle='-', label=f'Capacidad máxima: {max_load}')
    
    plt.title(f'Distribución de cargas por ruta - {result["instance"]}', fontsize=14)
    plt.xlabel('Ruta', fontsize=12)
    plt.ylabel('Carga', fontsize=12)
    plt.xticks(range(1, len(loads)+1))
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cargas_{result["instance"].replace(".dat", "")}.png', dpi=300)
    plt.show()
    
    # 2. Distribución de distancias por ruta
    plt.figure(figsize=(10, 6))
    distances = [route['distance'] for route in solution]
    
    plt.bar(range(1, len(distances)+1), distances, color='lightgreen')
    
    plt.title(f'Distribución de distancias por ruta - {result["instance"]}', fontsize=14)
    plt.xlabel('Ruta', fontsize=12)
    plt.ylabel('Distancia', fontsize=12)
    plt.xticks(range(1, len(distances)+1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'distancias_{result["instance"].replace(".dat", "")}.png', dpi=300)
    plt.show()
    
    # 3. Número de clientes por ruta
    plt.figure(figsize=(10, 6))
    customers_per_route = [len(route['customers']) for route in solution]
    
    plt.bar(range(1, len(customers_per_route)+1), customers_per_route, color='salmon')
    
    plt.title(f'Número de clientes por ruta - {result["instance"]}', fontsize=14)
    plt.xlabel('Ruta', fontsize=12)
    plt.ylabel('Número de clientes', fontsize=12)
    plt.xticks(range(1, len(customers_per_route)+1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'clientes_por_ruta_{result["instance"].replace(".dat", "")}.png', dpi=300)
    plt.show()

def generate_summary_table(results):
    """Genera una tabla de resumen con los resultados"""
    # Crear DataFrame con los resultados
    summary_data = []
    error_data = []
    
    for instance, data in results.items():
        if 'error' not in data:
            summary_data.append({
                'Instancia': instance,
                'Fitness': data['fitness'],
                'Generacion': data['generation'],
                'Poblacion': data['population_size'],
                'Tiempo (s)': data['execution_time']
            })
        else:
            error_data.append({
                'Instancia': instance,
                'Error': data['error']
            })
    
    # Si no hay resultados válidos
    if not summary_data:
        print("\nNo hay resultados válidos para mostrar en la tabla de resumen.")
        if error_data:
            print("\nInstancias con errores:")
            for err in error_data:
                print(f"{err['Instancia']}: {err['Error']}")
        return None
    
    df = pd.DataFrame(summary_data)
    
    # Ordenar por nombre de instancia
    df = df.sort_values('Instancia')
    
    # Imprimir la tabla
    print("\nResumen de resultados:")
    print("Instancia            Fitness    Generacion Poblacion  Tiempo (s)")
    print("----------------------------------------------------------------")
    
    for _, row in df.iterrows():
        print(f"{row['Instancia']:<20} {row['Fitness']:>10.2f} {row['Generacion']:>10d} {row['Poblacion']:>10d} {row['Tiempo (s)']:>10.2f}")
    
    # Imprimir instancias con errores
    if error_data:
        print("\nInstancias con errores:")
        for err in error_data:
            print(f"{err['Instancia']}: {err['Error']}")
    
    # Guardar también como CSV para análisis posterior
    df.to_csv('resumen_resultados.csv', index=False)
    
    # Guardar errores en un archivo separado si los hay
    if error_data:
        pd.DataFrame(error_data).to_csv('errores_instancias.csv', index=False)
    
    return df


def generate_visualizations(results):
    """Genera visualizaciones para el análisis de resultados"""
    # Comprobar si hay resultados válidos para visualizar
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        print("❌ No hay resultados válidos para visualizar.")
        return
    
    # 1. Gráfico de convergencia para cada instancia
    plt.figure(figsize=(15, 10))
    
    # Seleccionar algunas instancias representativas (máximo 10 para claridad)
    selected_instances = list(valid_results.keys())
    if len(selected_instances) > 10:
        # Seleccionar uniformemente 10 instancias
        step = len(selected_instances) // 10
        selected_instances = selected_instances[::step][:10]
    
    for instance in selected_instances:
        if 'error' not in results[instance]:
            convergence = results[instance]['convergence_history']
            plt.plot(convergence, label=instance)
    
    plt.title('Curvas de Convergencia BRKGA', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Distancia Total)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergencia_instancias.png', dpi=300)
    plt.show()
    
    # 2. Relación entre tamaño del problema y tiempo de ejecución
    problem_sizes = []
    execution_times = []
    fitness_values = []
    instance_names = []
    
    for instance, data in valid_results.items():
        try:
            if 'data' in data:
                mdvrp_data = data['data']
            else:
                # Si no hay datos en los resultados, cargar del archivo
                file_path = os.path.join('../dat', instance)
                mdvrp_data = parse_mdvrp_file(file_path)
                
            problem_size = mdvrp_data['num_customers'] + mdvrp_data['num_depots']
            
            problem_sizes.append(problem_size)
            execution_times.append(data['execution_time'])
            fitness_values.append(data['fitness'])
            instance_names.append(instance)
        except Exception as e:
            print(f"Error al procesar datos de {instance}: {str(e)}")
            pass
    
    # Verificar que hay datos para graficar
    if not problem_sizes:
        print("No hay suficientes datos para visualizar la relación tamaño vs tiempo.")
        return
    
    # Gráfico de dispersión: Tamaño vs Tiempo
    plt.figure(figsize=(10, 6))
    plt.scatter(problem_sizes, execution_times, alpha=0.7, s=80)
    
    for i, name in enumerate(instance_names):
        plt.annotate(name, (problem_sizes[i], execution_times[i]), 
                   fontsize=8, alpha=0.8)
    
    # Línea de tendencia
    z = np.polyfit(problem_sizes, execution_times, 1)
    p = np.poly1d(z)
    plt.plot(problem_sizes, p(problem_sizes), "r--", alpha=0.7)
    
    plt.title('Relación entre Tamaño del Problema y Tiempo de Ejecución', fontsize=14)
    plt.xlabel('Tamaño del Problema (Clientes + Depósitos)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tiempo_vs_tamaño.png', dpi=300)
    plt.show()
    
    # 3. Histograma de generaciones hasta convergencia
    plt.figure(figsize=(10, 6))
    
    generations = [data['generation'] for _, data in valid_results.items()]
    
    plt.hist(generations, bins=20, alpha=0.7, color='teal')
    plt.axvline(np.mean(generations), color='red', linestyle='dashed', linewidth=2, label=f'Media: {np.mean(generations):.1f}')
    
    plt.title('Distribución de Generaciones hasta Convergencia', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('histograma_convergencia.png', dpi=300)
    plt.show()
    
    # 4. Gráfico comparativo de fitness para todas las instancias
    plt.figure(figsize=(15, 8))
    
    df_summary = pd.DataFrame({
        'Instancia': instance_names,
        'Fitness': fitness_values,
        'Tiempo': execution_times
    })
    
    # Ordenar por fitness
    df_summary = df_summary.sort_values('Fitness')
    
    # Crear barplot
    ax = sns.barplot(x='Instancia', y='Fitness', data=df_summary)
    
    plt.title('Comparación de Fitness entre Instancias', fontsize=14)
    plt.xlabel('Instancia', fontsize=12)
    plt.ylabel('Fitness (Distancia Total)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparacion_fitness.png', dpi=300)
    plt.show()
    
    # 5. Mapa de calor: Generación de convergencia vs Tiempo
    if len(instance_names) > 5:
        plt.figure(figsize=(12, 10))
        
        # Crear DataFrame para el heatmap
        df_heatmap = pd.DataFrame({
            'Instancia': instance_names,
            'Generacion': [data['generation'] for _, data in valid_results.items() if 'instance' in data and data['instance'] in instance_names],
            'Tiempo': execution_times,
            'Fitness': fitness_values
        })
        
        # Seleccionar top 15 instancias por tiempo de ejecución para mejor visualización
        if len(df_heatmap) > 15:
            df_heatmap = df_heatmap.sort_values('Tiempo', ascending=False).head(15)
        
        # Crear matriz para el heatmap (Instancias vs Generación)
        heatmap_data = []
        for inst in df_heatmap['Instancia'].unique():
            row = {'Instancia': inst}
            inst_data = df_heatmap[df_heatmap['Instancia'] == inst]
            for gen in range(0, 1000, 100):  # Intervalos de 100 generaciones
                gen_key = f'Gen-{gen}'
                # Buscar generaciones en ese rango
                matching_gens = inst_data[(inst_data['Generacion'] >= gen) & 
                                        (inst_data['Generacion'] < gen + 100)]
                if len(matching_gens) > 0:
                    row[gen_key] = matching_gens['Tiempo'].mean()
                else:
                    row[gen_key] = 0
            heatmap_data.append(row)
        
        df_heat = pd.DataFrame(heatmap_data)
        df_heat = df_heat.set_index('Instancia')
        
        # Crear heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_heat, cmap="YlGnBu", annot=True, fmt=".1f", 
                   cbar_kws={'label': 'Tiempo de Ejecución (s)'})
        
        plt.title('Relación entre Instancias, Generaciones y Tiempo', fontsize=14)
        plt.xlabel('Rango de Generación de Convergencia', fontsize=12)
        plt.ylabel('Instancia', fontsize=12)
        plt.tight_layout()
        plt.savefig('heatmap_instancias_generacion.png', dpi=300)
        plt.show()
    
def analyze_early_stopping(results):
    """
    Analiza el impacto del early stopping adaptativo en los resultados.
    
    Args:
        results: Diccionario con los resultados de las ejecuciones
    """
    # Filtrar resultados válidos que contengan información de early stopping
    valid_results = {}
    for instance, data in results.items():
        if 'error' not in data and 'early_stopping' in data:
            valid_results[instance] = data
    
    if not valid_results:
        print("No hay suficientes datos para analizar el early stopping adaptativo.")
        return
    
    # Recopilar datos
    instances = []
    problem_sizes = []
    early_stop_values = []
    convergence_gens = []
    ratios = []  # Relación entre generación de convergencia y early stopping
    
    for instance, data in valid_results.items():
        try:
            # Cargar datos del problema
            if 'data' in data:
                mdvrp_data = data['data']
            else:
                # Si no hay datos, obtener del nombre de la instancia
                instances.append(instance)
                
            early_stop = data['early_stopping']
            conv_gen = data['generation']
            
            # Calcular tamaño del problema
            if 'data' in data:
                size = mdvrp_data['num_customers'] + mdvrp_data['num_depots']
            else:
                # Estimación aproximada basada en el nombre (para algunos formatos)
                parts = instance.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    size = int(parts[0]) + int(parts[1])
                else:
                    # Si no se puede determinar, usar un valor arbitrario
                    size = 100
            
            # Guardar datos
            problem_sizes.append(size)
            early_stop_values.append(early_stop)
            convergence_gens.append(conv_gen)
            ratios.append(conv_gen / early_stop if early_stop > 0 else 0)
            
        except Exception as e:
            print(f"Error procesando {instance}: {str(e)}")
    
    # Crear visualizaciones
    
    # 1. Relación entre tamaño del problema y early stopping
    plt.figure(figsize=(10, 6))
    plt.scatter(problem_sizes, early_stop_values, alpha=0.7, s=100)
    
    # Línea de tendencia
    z = np.polyfit(problem_sizes, early_stop_values, 1)
    p = np.poly1d(z)
    plt.plot(problem_sizes, p(problem_sizes), "r--", alpha=0.7)
    
    plt.title('Relación entre Tamaño del Problema y Early Stopping', fontsize=14)
    plt.xlabel('Tamaño del Problema (Clientes + Depósitos)', fontsize=12)
    plt.ylabel('Valor de Early Stopping', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tamaño_vs_early_stopping.png', dpi=300)
    plt.show()
    
    # 2. Relación entre early stopping y generación de convergencia
    plt.figure(figsize=(10, 6))
    plt.scatter(early_stop_values, convergence_gens, alpha=0.7, s=100)
    
    # Línea de tendencia
    z = np.polyfit(early_stop_values, convergence_gens, 1)
    p = np.poly1d(z)
    plt.plot(early_stop_values, p(early_stop_values), "r--", alpha=0.7)
    
    plt.title('Relación entre Early Stopping y Generación de Convergencia', fontsize=14)
    plt.xlabel('Valor de Early Stopping', fontsize=12)
    plt.ylabel('Generación de Convergencia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('early_stopping_vs_convergencia.png', dpi=300)
    plt.show()
    
    # 3. Distribución de la relación entre convergencia y early stopping
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=20, alpha=0.7, color='teal')
    plt.axvline(np.mean(ratios), color='red', linestyle='dashed', linewidth=2, 
               label=f'Media: {np.mean(ratios):.2f}')
    
    plt.title('Distribución de la Relación Convergencia/Early Stopping', fontsize=14)
    plt.xlabel('Convergencia / Early Stopping', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ratio_convergencia_early_stopping.png', dpi=300)
    plt.show()
    
    # Estadísticas descriptivas
    print("\n" + "="*80)
    print("ESTADÍSTICAS DE EARLY STOPPING ADAPTATIVO")
    print("="*80)
    print(f"Número de instancias analizadas: {len(problem_sizes)}")
    print(f"Valor promedio de early stopping: {np.mean(early_stop_values):.2f}")
    print(f"Valor mínimo de early stopping: {min(early_stop_values)}")
    print(f"Valor máximo de early stopping: {max(early_stop_values)}")
    print(f"Generación promedio de convergencia: {np.mean(convergence_gens):.2f}")
    print(f"Relación promedio Convergencia/Early Stopping: {np.mean(ratios):.2f}")
    
    # Análisis de correlación
    correlation = np.corrcoef(problem_sizes, early_stop_values)[0, 1]
    print(f"Correlación entre tamaño del problema y early stopping: {correlation:.2f}")
    
    correlation = np.corrcoef(early_stop_values, convergence_gens)[0, 1]
    print(f"Correlación entre early stopping y generación de convergencia: {correlation:.2f}")
    
    # Recomendar ajustes al factor base
    avg_ratio = np.mean(ratios)
    if avg_ratio < 0.3:
        print("\nRECOMENDACIÓN: El valor de early stopping parece ser demasiado alto en promedio,")
        print("considere reducir el factor base (actualmente 1000) para mejorar la eficiencia.")
    elif avg_ratio > 0.7:
        print("\nRECOMENDACIÓN: El valor de early stopping parece ser demasiado bajo en promedio,")
        print("considere aumentar el factor base (actualmente 1000) para mejorar la calidad de las soluciones.")
    else:
        print("\nRECOMENDACIÓN: El valor de early stopping parece estar bien ajustado en promedio.")
    
    return {
        'problem_sizes': problem_sizes,
        'early_stop_values': early_stop_values,
        'convergence_gens': convergence_gens,
        'ratios': ratios,
        'correlation_size_stop': correlation
    }    
# ----------------- FUNCIONES DE PARSEO Y UTILIDADES -----------------

def parse_mdvrp_file(file_path):
    """
    Parsea un archivo de instancia MDVRP y devuelve un diccionario estructurado con los datos.
    Soporta coordenadas de clientes como enteros o como flotantes.
    """
    data = {
        'problem_type': None,
        'num_vehicles': None,
        'num_customers': None,
        'num_depots': None,
        'depots': [],
        'customers': [],
        'vehicle_info': []
    }

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        
        # Procesar primera línea (tipo, m, n, t)
        first_line = lines[0].split()
        data['problem_type'] = int(first_line[0])
        data['num_vehicles'] = int(first_line[1])
        data['num_customers'] = int(first_line[2])
        data['num_depots'] = int(first_line[3])
        
        # Procesar información de vehículos/depósitos (próximas t líneas)
        for i in range(1, data['num_depots'] + 1):
            if i >= len(lines):
                break
            parts = lines[i].split()
            D, Q = float(parts[0]), float(parts[1])
            data['vehicle_info'].append({
                'max_duration': D,
                'max_load': Q,
                'depot_coords': None  # Se llenará después si hay información de depósitos
            })
        
        # Procesar clientes (el resto de líneas)
        customer_start = data['num_depots'] + 1
        for line in lines[customer_start:]:
            parts = line.split()
            if len(parts) < 6:  # Posiblemente líneas de depósitos al final
                continue
            
            try:
                # Intentar parsear con manejo de valores flotantes
                customer = {
                    'id': int(float(parts[0])),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'service_duration': float(parts[3]),
                    'demand': float(parts[4]),
                    'frequency': int(float(parts[5])),
                    'num_visit_combinations': int(float(parts[6])),
                    'visit_combinations': [int(float(x)) for x in parts[7:7+int(float(parts[6]))]],
                    'time_window': (float(parts[-2]), float(parts[-1])) if len(parts) > 7+int(float(parts[6])) else None
                }
                data['customers'].append(customer)
            except Exception as e:
                print(f"Warning: Error al parsear línea de cliente: {line}")
                print(f"Error: {str(e)}")
                # Continuar con la siguiente línea
                continue
        
        # En algunos archivos, los depósitos vienen después de los clientes
        # Buscamos líneas con demanda 0 que podrían ser depósitos
        for customer in data['customers']:
            if customer['demand'] == 0 and customer['frequency'] == 0:
                data['depots'].append({
                    'id': customer['id'],
                    'x': customer['x'],
                    'y': customer['y']
                })
        
        # Eliminar depósitos de la lista de clientes
        data['customers'] = [c for c in data['customers'] if c['demand'] > 0]
    
    # Verificar que se hayan cargado depósitos
    if not data['depots']:
        # Si no hay depósitos, intentar buscar al inicio del archivo
        print(f"No se encontraron depósitos automáticamente. Buscando en sección específica...")
        
        # Algunos archivos tienen los depósitos justo después de la información de vehículos
        depot_start = data['num_depots'] + 1
        depot_end = depot_start + data['num_depots']
        
        with open(file_path, 'r') as file:
            all_lines = file.readlines()
            
            # Extraer las líneas potenciales de depósitos
            if depot_end < len(all_lines):
                potential_depot_lines = all_lines[depot_start:depot_end]
                
                for i, line in enumerate(potential_depot_lines):
                    parts = line.strip().split()
                    if len(parts) >= 3:  # Id, X, Y como mínimo
                        try:
                            depot_id = int(float(parts[0]))
                            x = float(parts[1])
                            y = float(parts[2])
                            
                            data['depots'].append({
                                'id': depot_id,
                                'x': x,
                                'y': y
                            })
                            print(f"  Depósito añadido manualmente: ID={depot_id}, X={x}, Y={y}")
                        except Exception as e:
                            print(f"  Error al parsear depósito: {str(e)}")
    
    return data

def calculate_route_distance(depot, customers, data):
    """Calcula la distancia total de una ruta"""
    if not customers:
        return 0
    
    # Coordenadas del depósito
    depot_x, depot_y = depot['x'], depot['y']
    
    # Obtener coordenadas de todos los puntos en la ruta
    points = [(depot_x, depot_y)]
    for cust_id in customers:
        customer = next(c for c in data['customers'] if c['id'] == cust_id)
        points.append((customer['x'], customer['y']))
    points.append((depot_x, depot_y))  # Regreso al depósito
    
    # Calcular distancia euclidiana acumulada
    distance = 0
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        distance += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    return distance

# ----------------- FUNCIONES DE VISUALIZACIÓN -----------------

def plot_mdvrp_instance(data, show_demand=True, show_time_windows=False):
    """
    Visualiza una instancia MDVRP mostrando depósitos, clientes y demandas.
    
    Parámetros:
    - data: Diccionario con los datos parseados
    - show_demand: Si True, muestra el tamaño de los puntos según la demanda
    - show_time_windows: Si True, muestra información de ventanas de tiempo
    """
    plt.figure(figsize=(12, 8))
    
    # Configuración de colores
    depot_color = 'red'
    customer_color = 'blue'
    colors = cm.rainbow(np.linspace(0, 1, len(data['depots'])))
    
    # Dibujar depósitos
    for i, depot in enumerate(data['depots']):
        plt.scatter(depot['x'], depot['y'], 
                   c=[colors[i]], 
                   s=200, marker='s', 
                   edgecolors='black',
                   label=f'Depósito {depot["id"]}',
                   zorder=5)
    
    # Dibujar clientes
    demands = [c['demand'] for c in data['customers']]
    min_demand, max_demand = min(demands), max(demands)
    
    for customer in data['customers']:
        size = 50 + 150 * (customer['demand'] - min_demand) / (max_demand - min_demand) if show_demand else 50
        plt.scatter(customer['x'], customer['y'], 
                   c=customer_color, 
                   s=size, 
                   alpha=0.7,
                   edgecolors='black',
                   zorder=3)
        
        # Mostrar demanda como texto
        if show_demand:
            plt.text(customer['x'], customer['y'], 
                    str(int(customer['demand'])), 
                    fontsize=8, ha='center', va='center')
        
        # Mostrar ventanas de tiempo si está activado
        if show_time_windows and customer['time_window']:
            e, l = customer['time_window']
            plt.text(customer['x'], customer['y']-2, 
                    f'[{e}-{l}]', 
                    fontsize=7, ha='center', va='top', color='green')
    
    # Añadir detalles del gráfico
    plt.title(f'Instancia MDVRP - {len(data["depots"])} Depósitos, {len(data["customers"])} Clientes')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ajustar el zoom para ver todos los puntos
    all_x = [d['x'] for d in data['depots']] + [c['x'] for c in data['customers']]
    all_y = [d['y'] for d in data['depots']] + [c['y'] for c in data['customers']]
    plt.xlim(min(all_x)-5, max(all_x)+5)
    plt.ylim(min(all_y)-5, max(all_y)+5)
    
    plt.tight_layout()
    plt.show()

def visualize_routes(data, routes, show_demand=True, show_route_info=True):
    """
    Visualiza las rutas de la solución MDVRP
    
    Args:
        data: Diccionario con los datos del problema
        routes: Lista de rutas obtenidas de la solución BRKGA
        show_demand: Muestra el tamaño de los puntos según la demanda
        show_route_info: Muestra información sobre cada ruta
    """
    plt.figure(figsize=(14, 10))
    
    # Configuración de colores
    depot_color = 'red'
    customer_color = 'blue'
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    
    # Dibujar depósitos
    for depot in data['depots']:
        plt.scatter(depot['x'], depot['y'], 
                   c=depot_color, 
                   s=300, marker='s', 
                   edgecolors='black',
                   linewidths=2,
                   label='Depósito' if depot == data['depots'][0] else "",
                   zorder=5)
    
    # Dibujar clientes (todos primero)
    demands = [c['demand'] for c in data['customers']]
    min_demand, max_demand = min(demands), max(demands)
    
    for customer in data['customers']:
        size = 50 + 150 * (customer['demand'] - min_demand) / (max_demand - min_demand) if show_demand else 50
        plt.scatter(customer['x'], customer['y'], 
                   c=customer_color, 
                   s=size, 
                   alpha=0.7,
                   edgecolors='black',
                   zorder=3)
        
        if show_demand:
            plt.text(customer['x'], customer['y'], 
                    str(int(customer['demand'])), 
                    fontsize=9, ha='center', va='center', color='white')
    
    # Dibujar las rutas
    for i, route in enumerate(routes):
        # Obtener coordenadas de todos los puntos en la ruta
        depot = next(d for d in data['depots'] if d['id'] == route['depot_id'])
        points = [(depot['x'], depot['y'])]
        
        for cust_id in route['customers']:
            customer = next(c for c in data['customers'] if c['id'] == cust_id)
            points.append((customer['x'], customer['y']))
        
        points.append((depot['x'], depot['y']))  # Regreso al depósito
        
        # Separar coordenadas X e Y
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Dibujar la ruta
        plt.plot(x_coords, y_coords, 
                color=colors[i], 
                linestyle='-', 
                linewidth=2.5,
                marker='o',
                markersize=8 if show_route_info else 6,
                markerfacecolor='white' if show_route_info else colors[i],
                markeredgecolor=colors[i],
                markeredgewidth=1,
                alpha=0.9,
                label=f'Ruta {i+1}: {len(route["customers"])} clientes, Dist: {route["distance"]:.1f}',
                zorder=4)
        
        # Marcar el orden de visita
        if show_route_info:
            for j in range(1, len(points)-1):
                plt.text(points[j][0], points[j][1], 
                        str(j), 
                        fontsize=8, ha='center', va='center', 
                        color='black', weight='bold')
    
    # Añadir detalles del gráfico
    plt.title('Solución MDVRP - Visualización de Rutas', fontsize=14, pad=20)
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Leyenda mejorada
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title('Leyenda', prop={'size': 11, 'weight': 'bold'})
    
    # Ajustar el zoom para ver todos los puntos
    all_x = [d['x'] for d in data['depots']] + [c['x'] for c in data['customers']]
    all_y = [d['y'] for d in data['depots']] + [c['y'] for c in data['customers']]
    plt.xlim(min(all_x)-10, max(all_x)+10)
    plt.ylim(min(all_y)-10, max(all_y)+10)
    
    # Añadir información general
    plt.figtext(0.5, 0.01, 
                f"Total Rutas: {len(routes)} | Distancia Total: {sum(r['distance'] for r in routes):.1f} | " +
                f"Clientes Atendidos: {sum(len(r['customers']) for r in routes)}/{len(data['customers'])}", 
                ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    plt.tight_layout()
    plt.show()
    
def analyze_refinement_impact(data_dir, instance_file, population_size=100, elite_percent=0.1, mutants_percent=0.1, max_generations=200):
    """
    Analiza el impacto del refinamiento ejecutando el algoritmo con y sin refinamiento
    y comparando los resultados.
    
    Args:
        data_dir: Directorio con los datos
        instance_file: Archivo de instancia a analizar
        population_size, elite_percent, mutants_percent, max_generations: Parámetros del algoritmo
    """
    file_path = os.path.join(data_dir, instance_file)
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE IMPACTO DEL REFINAMIENTO - {instance_file}")
    print(f"{'='*80}")
    
    try:
        # Cargar datos de la instancia
        mdvrp_data = parse_mdvrp_file(file_path)
        print(f"Instancia cargada: {mdvrp_data['num_customers']} clientes, {mdvrp_data['num_depots']} depósitos")
        
        # Calcular early stopping adaptativo
        early_stop = calculate_early_stopping(mdvrp_data['num_customers'], mdvrp_data['num_depots'])
        print(f"Early stopping adaptativo: {early_stop} generaciones sin mejora")
        
        results = {}
        
        # Ejecutar sin refinamiento
        print("\nEjecutando BRKGA sin refinamiento...")
        start_time = time.time()
        brkga_no_refine = BRKGA_MDVRP(
            mdvrp_data, 
            population_size=population_size, 
            elite_percent=elite_percent, 
            mutants_percent=mutants_percent,
            use_refinement=False
        )
        
        solution_no_refine, distance_no_refine, chromosome_no_refine, history_no_refine, convergence_no_refine = brkga_no_refine.solve(
            generations=max_generations, 
            verbose=True,
            early_stopping=early_stop
        )
        
        time_no_refine = time.time() - start_time
        results['no_refinement'] = {
            'solution': solution_no_refine,
            'distance': distance_no_refine,
            'execution_time': time_no_refine,
            'convergence_gen': convergence_no_refine,
            'history': history_no_refine
        }
        
        # Ejecutar con refinamiento
        print("\nEjecutando BRKGA con refinamiento...")
        start_time = time.time()
        brkga_refine = BRKGA_MDVRP(
            mdvrp_data, 
            population_size=population_size, 
            elite_percent=elite_percent, 
            mutants_percent=mutants_percent,
            use_refinement=True
        )
        
        solution_refine, distance_refine, chromosome_refine, history_refine, convergence_refine = brkga_refine.solve(
            generations=max_generations, 
            verbose=True,
            early_stopping=early_stop
        )
        
        time_refine = time.time() - start_time
        results['refinement'] = {
            'solution': solution_refine,
            'distance': distance_refine,
            'execution_time': time_refine,
            'convergence_gen': convergence_refine,
            'history': history_refine
        }
        
        # Validar soluciones
        print("\nValidando soluciones...")
        print("\nSolución sin refinamiento:")
        valid_no_refine, violations_no_refine = debug_solution(mdvrp_data, solution_no_refine, verbose=False)
        print(f"  Válida: {'✅ Sí' if valid_no_refine else '❌ No'}")
        
        print("\nSolución con refinamiento:")
        valid_refine, violations_refine = debug_solution(mdvrp_data, solution_refine, verbose=False)
        print(f"  Válida: {'✅ Sí' if valid_refine else '❌ No'}")
        
        # Comparar resultados
        improvement = ((distance_no_refine - distance_refine) / distance_no_refine) * 100
        time_increase = ((time_refine - time_no_refine) / time_no_refine) * 100
        
        print("\n" + "="*80)
        print("RESULTADOS COMPARATIVOS")
        print("="*80)
        print(f"Early stopping adaptativo: {early_stop}")
        print(f"Distancia sin refinamiento: {distance_no_refine:.2f}")
        print(f"Distancia con refinamiento: {distance_refine:.2f}")
        print(f"Mejora: {improvement:.2f}%")
        print(f"Tiempo sin refinamiento: {time_no_refine:.2f}s")
        print(f"Tiempo con refinamiento: {time_refine:.2f}s")
        print(f"Incremento en tiempo: {time_increase:.2f}%")
        print(f"Generación de convergencia sin refinamiento: {convergence_no_refine}")
        print(f"Generación de convergencia con refinamiento: {convergence_refine}")
        
        
        # Visualizar curvas de convergencia
        plt.figure(figsize=(12, 6))
        plt.plot(history_no_refine, label='Sin refinamiento', color='blue')
        plt.plot(history_refine, label='Con refinamiento', color='red')
        plt.axvline(x=convergence_no_refine, color='blue', linestyle='--', 
                   label=f'Convergencia sin refinamiento (Gen {convergence_no_refine})')
        plt.axvline(x=convergence_refine, color='red', linestyle='--', 
                   label=f'Convergencia con refinamiento (Gen {convergence_refine})')
        
        plt.title(f'Comparación de convergencia con/sin refinamiento - {instance_file}')
        plt.xlabel('Generación')
        plt.ylabel('Fitness (Distancia)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'comparacion_refinamiento_{instance_file.replace(".dat", "")}.png', dpi=300)
        plt.show()
        
        # Visualizar soluciones
        print("\nVisualizando solución sin refinamiento...")
        solution_no_refine_with_data = {
            'best_solution': solution_no_refine,
            'data': mdvrp_data,
            'fitness': distance_no_refine
        }
        visualize_routes(mdvrp_data, solution_no_refine, show_demand=True, show_route_info=True)
        
        print("\nVisualizando solución con refinamiento...")
        solution_refine_with_data = {
            'best_solution': solution_refine,
            'data': mdvrp_data,
            'fitness': distance_refine
        }
        visualize_routes(mdvrp_data, solution_refine, show_demand=True, show_route_info=True)
        
        # Análisis estadístico
        
        # 1. Comparar número de rutas
        num_routes_no_refine = len(solution_no_refine)
        num_routes_refine = len(solution_refine)
        
        print("\nEstadísticas:")
        print(f"  Número de rutas sin refinamiento: {num_routes_no_refine}")
        print(f"  Número de rutas con refinamiento: {num_routes_refine}")
        
        # 2. Análisis de distribución de clientes por ruta
        customers_per_route_no_refine = [len(r['customers']) for r in solution_no_refine]
        customers_per_route_refine = [len(r['customers']) for r in solution_refine]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(customers_per_route_no_refine, bins=range(1, max(customers_per_route_no_refine)+2), 
                alpha=0.7, color='blue')
        plt.title('Clientes por ruta - Sin refinamiento')
        plt.xlabel('Número de clientes')
        plt.ylabel('Número de rutas')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(customers_per_route_refine, bins=range(1, max(customers_per_route_refine)+2), 
                alpha=0.7, color='red')
        plt.title('Clientes por ruta - Con refinamiento')
        plt.xlabel('Número de clientes')
        plt.ylabel('Número de rutas')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'clientes_por_ruta_comparacion_{instance_file.replace(".dat", "")}.png', dpi=300)
        plt.show()
        
        # 3. Análisis de balance de carga entre depósitos
        depots_no_refine = {}
        depots_refine = {}
        
        for route in solution_no_refine:
            depot_id = route['depot_id']
            if depot_id not in depots_no_refine:
                depots_no_refine[depot_id] = {'routes': 0, 'customers': 0, 'load': 0}
            depots_no_refine[depot_id]['routes'] += 1
            depots_no_refine[depot_id]['customers'] += len(route['customers'])
            depots_no_refine[depot_id]['load'] += route['load']
        
        for route in solution_refine:
            depot_id = route['depot_id']
            if depot_id not in depots_refine:
                depots_refine[depot_id] = {'routes': 0, 'customers': 0, 'load': 0}
            depots_refine[depot_id]['routes'] += 1
            depots_refine[depot_id]['customers'] += len(route['customers'])
            depots_refine[depot_id]['load'] += route['load']
        
        # Visualizar balance de depósitos
        depot_ids = sorted(set(depots_no_refine.keys()) | set(depots_refine.keys()))
        
        # Crear gráfico de barras para comparar rutas por depósito
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Rutas por depósito
        routes_no_refine = [depots_no_refine.get(d, {'routes': 0})['routes'] for d in depot_ids]
        routes_refine = [depots_refine.get(d, {'routes': 0})['routes'] for d in depot_ids]
        
        x = np.arange(len(depot_ids))
        width = 0.35
        
        ax1.bar(x - width/2, routes_no_refine, width, label='Sin refinamiento', color='blue', alpha=0.7)
        ax1.bar(x + width/2, routes_refine, width, label='Con refinamiento', color='red', alpha=0.7)
        ax1.set_title('Número de rutas por depósito')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Depósito {d}' for d in depot_ids])
        ax1.set_ylabel('Número de rutas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Clientes por depósito
        customers_no_refine = [depots_no_refine.get(d, {'customers': 0})['customers'] for d in depot_ids]
        customers_refine = [depots_refine.get(d, {'customers': 0})['customers'] for d in depot_ids]
        
        ax2.bar(x - width/2, customers_no_refine, width, label='Sin refinamiento', color='blue', alpha=0.7)
        ax2.bar(x + width/2, customers_refine, width, label='Con refinamiento', color='red', alpha=0.7)
        ax2.set_title('Número de clientes por depósito')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Depósito {d}' for d in depot_ids])
        ax2.set_ylabel('Número de clientes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Carga por depósito
        load_no_refine = [depots_no_refine.get(d, {'load': 0})['load'] for d in depot_ids]
        load_refine = [depots_refine.get(d, {'load': 0})['load'] for d in depot_ids]
        
        ax3.bar(x - width/2, load_no_refine, width, label='Sin refinamiento', color='blue', alpha=0.7)
        ax3.bar(x + width/2, load_refine, width, label='Con refinamiento', color='red', alpha=0.7)
        ax3.set_title('Carga total por depósito')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Depósito {d}' for d in depot_ids])
        ax3.set_ylabel('Carga total')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'balance_depositos_{instance_file.replace(".dat", "")}.png', dpi=300)
        plt.show()
        
        return {
            'no_refinement': results['no_refinement'],
            'refinement': results['refinement'],
            'improvement_percentage': improvement,
            'time_increase_percentage': time_increase
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error en el análisis: {str(e)}")
        print(traceback.format_exc())
        return None

# ----------------- FUNCIONES DE VERIFICACIÓN Y DEPURACIÓN -----------------

def debug_solution(data, routes, verbose=True):
    """
    Verifica que una solución cumpla con todas las restricciones del MDVRP
    
    Args:
        data: Diccionario con los datos del problema
        routes: Lista de rutas de la solución a verificar
        verbose: Si True, muestra detalles de las verificaciones
    
    Returns:
        Tuple: (is_valid, violations) donde:
            is_valid: Booleano indicando si la solución es válida
            violations: Diccionario con el conteo de violaciones por tipo
    """
    violations = {
        'capacity': 0,
        'duration': 0,
        'unserved_customers': 0,
        'duplicate_customers': 0,
        'wrong_depot': 0,
        'time_windows': 0
    }
    
    # 1. Verificar que todos los clientes sean atendidos exactamente una vez
    served_customers = []
    for route in routes:
        served_customers.extend(route['customers'])
    
    all_customer_ids = [c['id'] for c in data['customers']]
    
    # Clientes no atendidos
    unserved = set(all_customer_ids) - set(served_customers)
    violations['unserved_customers'] = len(unserved)
    
    # Clientes duplicados
    duplicate_customers = [cid for cid in served_customers if served_customers.count(cid) > 1]
    violations['duplicate_customers'] = len(set(duplicate_customers))
    
    # 2. Verificar restricciones por ruta
    for i, route in enumerate(routes):
        if verbose:
            print(f"\nVerificando Ruta {i+1} (Depósito {route['depot_id']}):")
        
        # Obtener depósito correspondiente
        try:
            depot = next(d for d in data['depots'] if d['id'] == route['depot_id'])
        except StopIteration:
            if verbose:
                print(f"  ❌ ERROR: Depósito {route['depot_id']} no existe")
            violations['wrong_depot'] += 1
            continue
        
        # Calcular carga total y duración
        total_load = 0
        total_duration = 0
        current_time = 0  # Para verificación de ventanas de tiempo
        
        # Coordenadas iniciales (depósito)
        prev_x, prev_y = depot['x'], depot['y']
        
        for j, cust_id in enumerate(route['customers']):
            try:
                customer = next(c for c in data['customers'] if c['id'] == cust_id)
            except StopIteration:
                if verbose:
                    print(f"  ❌ ERROR: Cliente {cust_id} no existe")
                continue
            
            # Verificar carga
            total_load += customer['demand']
            
            # Calcular tiempo de viaje al cliente
            distance = np.sqrt((customer['x']-prev_x)**2 + (customer['y']-prev_y)**2)
            current_time += distance  # Tiempo de viaje
            
            # Verificar ventana de tiempo
            if customer['time_window']:
                e, l = customer['time_window']
                if current_time < e:
                    # Esperar hasta el inicio de la ventana
                    current_time = e
                elif current_time > l:
                    if verbose:
                        print(f"  ❌ Violación de ventana de tiempo en cliente {cust_id}: "
                              f"Llegada {current_time:.2f} > Fin {l}")
                    violations['time_windows'] += 1
            
            # Tiempo de servicio
            current_time += customer['service_duration']
            total_duration += distance + customer['service_duration']
            
            # Actualizar coordenadas para el siguiente cálculo
            prev_x, prev_y = customer['x'], customer['y']
        
        # Añadir el regreso al depósito
        distance_to_depot = np.sqrt((depot['x']-prev_x)**2 + (depot['y']-prev_y)**2)
        total_duration += distance_to_depot
        
        # Verificar capacidad máxima
        if total_load > data['vehicle_info'][0]['max_load']:
            if verbose:
                print(f"  ❌ Violación de capacidad: {total_load} > {data['vehicle_info'][0]['max_load']}")
            violations['capacity'] += 1
        
        # Verificar duración máxima
        if total_duration > data['vehicle_info'][0]['max_duration']:
            if verbose:
                print(f"  ❌ Violación de duración: {total_duration:.2f} > {data['vehicle_info'][0]['max_duration']}")
            violations['duration'] += 1
        
        if verbose:
            print(f"  ✔ Carga: {total_load}/{data['vehicle_info'][0]['max_load']}")
            print(f"  ✔ Duración: {total_duration:.2f}/{data['vehicle_info'][0]['max_duration']}")
    
    # 3. Resumen de verificaciones
    is_valid = all(v == 0 for v in violations.values())
    
    if verbose:
        print("\n" + "="*50)
        print(" RESUMEN DE VERIFICACIÓN:")
        print(f" - Clientes no atendidos: {violations['unserved_customers']}")
        print(f" - Clientes duplicados: {violations['duplicate_customers']}")
        print(f" - Rutas con depósito incorrecto: {violations['wrong_depot']}")
        print(f" - Violaciones de capacidad: {violations['capacity']}")
        print(f" - Violaciones de duración: {violations['duration']}")
        print(f" - Violaciones de ventanas de tiempo: {violations['time_windows']}")
        print("\n" + "="*50)
        print(f"SOLUCIÓN {'VÁLIDA' if is_valid else 'NO VÁLIDA'}")
        print("="*50)
    
    return is_valid, violations


def calculate_early_stopping(num_customers, num_depots, base_factor=1000, min_stop=20, max_stop=200):
    """
    Calcula un valor de early stopping adaptativo basado en el tamaño de la instancia.
    
    Args:
        num_customers: Número de clientes en la instancia
        num_depots: Número de depósitos en la instancia
        base_factor: Factor base para el cálculo (default: 1000)
        min_stop: Valor mínimo de early stopping (default: 20)
        max_stop: Valor máximo de early stopping (default: 200)
    
    Returns:
        int: Número de generaciones sin mejora para detener el algoritmo
    """
    # Calculamos el tamaño total del problema
    problem_size = num_customers + num_depots
    
    # Para problemas grandes, early stopping más pequeño
    # Para problemas pequeños, early stopping más grande
    early_stop = int(base_factor / problem_size)
    
    # Limitamos entre min_stop y max_stop
    return max(min_stop, min(early_stop, max_stop))

# ----------------- IMPLEMENTACIÓN DEL ALGORITMO BRKGA -----------------

class BRKGA_MDVRP:
    def __init__(self, data, population_size=100, elite_percent=0.2, mutants_percent=0.1, use_refinement=True):
        """
        Inicializa el BRKGA para MDVRP
        
        Args:
            data: Diccionario con los datos del problema
            population_size: Tamaño de la población
            elite_percent: Porcentaje de individuos élite
            mutants_percent: Porcentaje de mutantes
            use_refinement: Si True, aplica refinamiento a las soluciones
        """
        self.data = data
        self.population_size = population_size
        self.elite_size = int(population_size * elite_percent)
        self.mutants_size = int(population_size * mutants_percent)
        self.num_depots = len(data['depots'])
        self.use_refinement = use_refinement
        
        # Número de genes: 2 genes por cliente (asignación + orden)
        self.num_genes = 2 * len(data['customers'])
        
        # Parámetros de decodificación
        self.p_bias = 0.7  # Probabilidad de tomar gen del padre élite
        
    def initialize_population(self):
        """Inicializa la población aleatoriamente"""
        return np.random.rand(self.population_size, self.num_genes)
    
    def calculate_route_distance(self, depot, customers):
        """Calcula la distancia total de una ruta"""
        if not customers:
            return 0
        
        # Coordenadas del depósito
        depot_x, depot_y = depot['x'], depot['y']
        
        # Obtener coordenadas de todos los puntos en la ruta
        points = [(depot_x, depot_y)]  # Comienza en el depósito
        for cust_id in customers:
            customer = next(c for c in self.data['customers'] if c['id'] == cust_id)
            points.append((customer['x'], customer['y']))
        points.append((depot_x, depot_y))  # Regresa al depósito
        
        # Calcular distancia euclidiana acumulada
        distance = 0
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            distance += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        return distance

    def crossover(self, elite_parent, non_elite_parent):
        """Operador de cruce sesgado"""
        child = np.where(np.random.rand(self.num_genes) < self.p_bias, 
                         elite_parent, non_elite_parent)
        return child
    
    def evolve(self, population, fitness_values):
        """Evoluciona la población a la siguiente generación"""
        # Ordenar la población por fitness
        sorted_indices = np.argsort(fitness_values)
        elite = population[sorted_indices[:self.elite_size]]
        non_elite = population[sorted_indices[self.elite_size:]]
        
        # Generar descendencia
        offspring = []
        for _ in range(self.population_size - self.elite_size - self.mutants_size):
            elite_parent = elite[np.random.randint(self.elite_size)]
            non_elite_parent = non_elite[np.random.randint(len(non_elite))]
            offspring.append(self.crossover(elite_parent, non_elite_parent))
        
        # Crear mutantes
        mutants = np.random.rand(self.mutants_size, self.num_genes)
        
        # Nueva población = élite + descendencia + mutantes
        new_population = np.vstack([elite, offspring, mutants])
        
        return new_population
    
    def refine_solution(self, routes):
        """
        Refina una solución aplicando técnicas de búsqueda local.
        
        Args:
            routes: Lista de rutas a refinar
            
        Returns:
            routes: Rutas refinadas
            total_distance: Distancia total refinada
        """
        # Definimos las técnicas de refinamiento a aplicar
        refinement_techniques = [
            self._apply_2opt_intraroute,
            self._apply_relocation_intraroute,
            self._apply_relocation_interroute,
            self._apply_swap_interroute
        ]
        
        improved = True
        iteration = 0
        max_iterations = 5  # Limitar el número de iteraciones para evitar tiempos excesivos
        
        # Calcular distancia inicial
        total_distance = sum(route['distance'] for route in routes)
        
        while improved and iteration < max_iterations:
            improved = False
            initial_distance = total_distance
            
            # Aplicar cada técnica de refinamiento
            for technique in refinement_techniques:
                routes, total_distance = technique(routes, total_distance)
            
            # Verificar si hubo mejora
            if total_distance < initial_distance:
                improved = True
                
            iteration += 1
        
        return routes, total_distance

    def _apply_2opt_intraroute(self, routes, current_distance):
        """
        Aplica el movimiento 2-opt dentro de cada ruta.
        Intercambia dos arcos no adyacentes dentro de una ruta para eliminar cruces.
        """
        improved_routes = []
        total_distance = 0
        
        for route in routes:
            depot_id = route['depot_id']
            customers = route['customers'].copy()
            depot = next(d for d in self.data['depots'] if d['id'] == depot_id)
            
            if len(customers) <= 2:
                # No se puede aplicar 2-opt con menos de 3 clientes
                improved_routes.append(route)
                total_distance += route['distance']
                continue
            
            # Aplicar 2-opt
            best_distance = route['distance']
            best_customers = customers.copy()
            
            for i in range(len(customers) - 1):
                for j in range(i + 2, len(customers)):
                    if j - i == 1:
                        continue  # Arcos adyacentes, no aplicar 2-opt
                    
                    # Invertir el segmento entre i+1 y j
                    new_customers = customers.copy()
                    new_customers[i+1:j+1] = reversed(customers[i+1:j+1])
                    
                    # Calcular nueva distancia
                    new_distance = self.calculate_route_distance(depot, new_customers)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_customers = new_customers.copy()
            
            # Actualizar ruta
            improved_route = {
                'depot_id': depot_id,
                'customers': best_customers,
                'load': route['load'],
                'distance': best_distance
            }
            
            improved_routes.append(improved_route)
            total_distance += best_distance
        
        return improved_routes, total_distance

    def _apply_relocation_intraroute(self, routes, current_distance):
        """
        Mueve un cliente a otra posición dentro de la misma ruta.
        """
        improved_routes = []
        total_distance = 0
        
        for route in routes:
            depot_id = route['depot_id']
            customers = route['customers'].copy()
            depot = next(d for d in self.data['depots'] if d['id'] == depot_id)
            
            if len(customers) <= 1:
                improved_routes.append(route)
                total_distance += route['distance']
                continue
            
            # Aplicar relocation
            best_distance = route['distance']
            best_customers = customers.copy()
            
            for i in range(len(customers)):
                customer_to_move = customers[i]
                
                # Probar todas las posiciones posibles
                for j in range(len(customers) + 1):
                    if j == i or j == i + 1:
                        continue  # Misma posición o posición adyacente, no cambio
                    
                    # Crear nueva secuencia moviendo el cliente
                    new_customers = customers.copy()
                    new_customers.pop(i)
                    if j > i:
                        j -= 1  # Ajustar índice después de eliminar
                    new_customers.insert(j, customer_to_move)
                    
                    # Calcular nueva distancia
                    new_distance = self.calculate_route_distance(depot, new_customers)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_customers = new_customers.copy()
            
            # Actualizar ruta
            improved_route = {
                'depot_id': depot_id,
                'customers': best_customers,
                'load': route['load'],
                'distance': best_distance
            }
            
            improved_routes.append(improved_route)
            total_distance += best_distance
        
        return improved_routes, total_distance

    def _apply_relocation_interroute(self, routes, current_distance):
        """
        Mueve un cliente de una ruta a otra, verificando restricciones de capacidad.
        """
        if len(routes) <= 1:
            return routes, current_distance
        
        improved = True
        current_routes = routes.copy()
        
        while improved:
            improved = False
            best_distance = sum(route['distance'] for route in current_routes)
            best_routes = current_routes.copy()
            
            # Intentar mover cada cliente de cada ruta a cada otra ruta
            for i, source_route in enumerate(current_routes):
                depot_i = next(d for d in self.data['depots'] if d['id'] == source_route['depot_id'])
                
                if not source_route['customers']:
                    continue
                    
                for cust_idx, customer_id in enumerate(source_route['customers']):
                    customer = next(c for c in self.data['customers'] if c['id'] == customer_id)
                    
                    for j, target_route in enumerate(current_routes):
                        if i == j:
                            continue
                            
                        depot_j = next(d for d in self.data['depots'] if d['id'] == target_route['depot_id'])
                        
                        # Verificar restricción de capacidad
                        if target_route['load'] + customer['demand'] > self.data['vehicle_info'][0]['max_load']:
                            continue
                        
                        # Probar todas las posiciones en la ruta destino
                        for pos in range(len(target_route['customers']) + 1):
                            # Crear rutas temporales
                            new_routes = []
                            
                            # Ruta origen sin el cliente
                            new_source_customers = source_route['customers'].copy()
                            new_source_customers.pop(cust_idx)
                            
                            new_source_route = {
                                'depot_id': source_route['depot_id'],
                                'customers': new_source_customers,
                                'load': source_route['load'] - customer['demand'],
                                'distance': 0  # Se calculará después
                            }
                            
                            # Ruta destino con el cliente añadido
                            new_target_customers = target_route['customers'].copy()
                            new_target_customers.insert(pos, customer_id)
                            
                            new_target_route = {
                                'depot_id': target_route['depot_id'],
                                'customers': new_target_customers,
                                'load': target_route['load'] + customer['demand'],
                                'distance': 0  # Se calculará después
                            }
                            
                            # Actualizar distancias
                            new_source_route['distance'] = self.calculate_route_distance(depot_i, new_source_route['customers'])
                            new_target_route['distance'] = self.calculate_route_distance(depot_j, new_target_route['customers'])
                            
                            # Crear conjunto completo de rutas
                            for k, route in enumerate(current_routes):
                                if k == i:
                                    new_routes.append(new_source_route)
                                elif k == j:
                                    new_routes.append(new_target_route)
                                else:
                                    new_routes.append(route)
                            
                            # Calcular distancia total
                            new_total_distance = sum(r['distance'] for r in new_routes)
                            
                            if new_total_distance < best_distance:
                                best_distance = new_total_distance
                                best_routes = new_routes.copy()
                                improved = True
            
            if improved:
                current_routes = best_routes.copy()
        
        return current_routes, sum(route['distance'] for route in current_routes)

    def _apply_swap_interroute(self, routes, current_distance):
        """
        Intercambia un cliente de una ruta con un cliente de otra ruta.
        """
        if len(routes) <= 1:
            return routes, current_distance
        
        improved = True
        current_routes = routes.copy()
        
        while improved:
            improved = False
            best_distance = sum(route['distance'] for route in current_routes)
            best_routes = current_routes.copy()
            
            # Intentar intercambiar cada par de clientes entre rutas
            for i, route_i in enumerate(current_routes):
                if not route_i['customers']:
                    continue
                    
                depot_i = next(d for d in self.data['depots'] if d['id'] == route_i['depot_id'])
                
                for j, route_j in enumerate(current_routes):
                    if i == j or not route_j['customers']:
                        continue
                        
                    depot_j = next(d for d in self.data['depots'] if d['id'] == route_j['depot_id'])
                    
                    for idx_i, cust_i in enumerate(route_i['customers']):
                        customer_i = next(c for c in self.data['customers'] if c['id'] == cust_i)
                        
                        for idx_j, cust_j in enumerate(route_j['customers']):
                            customer_j = next(c for c in self.data['customers'] if c['id'] == cust_j)
                            
                            # Verificar restricciones de capacidad después del intercambio
                            new_load_i = route_i['load'] - customer_i['demand'] + customer_j['demand']
                            new_load_j = route_j['load'] - customer_j['demand'] + customer_i['demand']
                            
                            if (new_load_i > self.data['vehicle_info'][0]['max_load'] or 
                                new_load_j > self.data['vehicle_info'][0]['max_load']):
                                continue
                            
                            # Crear rutas temporales con el intercambio
                            new_routes = []
                            
                            # Ruta i con el cliente de j
                            new_route_i_customers = route_i['customers'].copy()
                            new_route_i_customers[idx_i] = cust_j
                            
                            new_route_i = {
                                'depot_id': route_i['depot_id'],
                                'customers': new_route_i_customers,
                                'load': new_load_i,
                                'distance': 0  # Se calculará después
                            }
                            
                            # Ruta j con el cliente de i
                            new_route_j_customers = route_j['customers'].copy()
                            new_route_j_customers[idx_j] = cust_i
                            
                            new_route_j = {
                                'depot_id': route_j['depot_id'],
                                'customers': new_route_j_customers,
                                'load': new_load_j,
                                'distance': 0  # Se calculará después
                            }
                            
                            # Actualizar distancias
                            new_route_i['distance'] = self.calculate_route_distance(depot_i, new_route_i['customers'])
                            new_route_j['distance'] = self.calculate_route_distance(depot_j, new_route_j['customers'])
                            
                            # Crear conjunto completo de rutas
                            for k, route in enumerate(current_routes):
                                if k == i:
                                    new_routes.append(new_route_i)
                                elif k == j:
                                    new_routes.append(new_route_j)
                                else:
                                    new_routes.append(route)
                            
                            # Calcular distancia total
                            new_total_distance = sum(r['distance'] for r in new_routes)
                            
                            if new_total_distance < best_distance:
                                best_distance = new_total_distance
                                best_routes = new_routes.copy()
                                improved = True
            
            if improved:
                current_routes = best_routes.copy()
        
        return current_routes, sum(route['distance'] for route in current_routes)
    
    def decode(self, chromosome):
        """
        Decodifica un cromosoma en una solución factible y aplica refinamiento
        
        Returns:
            solution: Lista de rutas por depósito
            total_distance: Distancia total de la solución
        """
        num_customers = len(self.data['customers'])
    
        # 1. Asignar clientes a depósitos
        depot_assignments = []
        for i in range(num_customers):
            depot_idx = int(chromosome[i] * self.num_depots)
            depot_idx = min(depot_idx, self.num_depots-1)
            depot_assignments.append(depot_idx)
        
        # 2. Ordenar clientes dentro de cada depósito
        sorted_indices = np.argsort(chromosome[num_customers:])
        
        # 3. Construir rutas iniciales
        routes = {depot['id']: [] for depot in self.data['depots']}
        for idx in sorted_indices:
            depot_id = self.data['depots'][depot_assignments[idx]]['id']
            customer_id = self.data['customers'][idx]['id']
            routes[depot_id].append(customer_id)
        
        # 4. Dividir en rutas factibles
        feasible_routes = []
        total_distance = 0
        
        for depot_id, customers in routes.items():
            depot = next(d for d in self.data['depots'] if d['id'] == depot_id)
            current_route = []
            current_load = 0
            current_duration = 0
            
            for cust_id in customers:
                customer = next(c for c in self.data['customers'] if c['id'] == cust_id)
                
                if (current_load + customer['demand'] > self.data['vehicle_info'][0]['max_load'] or
                    current_duration + customer['service_duration'] > self.data['vehicle_info'][0]['max_duration']):
                    if current_route:
                        route_distance = self.calculate_route_distance(depot, current_route)
                        feasible_routes.append({
                            'depot_id': depot_id,
                            'customers': current_route,
                            'load': current_load,
                            'distance': route_distance
                        })
                        total_distance += route_distance
                    
                    current_route = []
                    current_load = 0
                    current_duration = 0
                
                current_route.append(cust_id)
                current_load += customer['demand']
                current_duration += customer['service_duration']
            
            if current_route:
                route_distance = self.calculate_route_distance(depot, current_route)
                feasible_routes.append({
                    'depot_id': depot_id,
                    'customers': current_route,
                    'load': current_load,
                    'distance': route_distance
                })
                total_distance += route_distance
        
        # 5. Aplicar refinamiento a la solución si está habilitado
        if self.use_refinement:
            refined_routes, refined_distance = self.refine_solution(feasible_routes)
            return refined_routes, refined_distance
        else:
            return feasible_routes, total_distance
        
    def fitness(self, chromosome):
        """Evalúa la calidad de un cromosoma considerando múltiples factores"""
        solution, total_distance = self.decode(chromosome)
        
        # Factores a considerar en la evaluación
        num_routes = len(solution)
        used_depots = set(route['depot_id'] for route in solution)
        num_used_depots = len(used_depots)
        
        # Penalización por no usar todos los depósitos
        depot_penalty = 1000 * (self.num_depots - num_used_depots) if num_used_depots < self.num_depots else 0
        
        # Penalización por número excesivo de vehículos
        vehicle_penalty = 100 * max(0, num_routes - self.data['num_vehicles'])
        
        # Penalización por desequilibrio en la carga de los depósitos
        if num_used_depots > 0:
            routes_per_depot = {}
            for route in solution:
                depot_id = route['depot_id']
                if depot_id not in routes_per_depot:
                    routes_per_depot[depot_id] = 0
                routes_per_depot[depot_id] += 1
            
            # Calcular desviación estándar del número de rutas por depósito
            mean_routes = sum(routes_per_depot.values()) / len(routes_per_depot)
            variance = sum((v - mean_routes) ** 2 for v in routes_per_depot.values()) / len(routes_per_depot)
            std_dev = variance ** 0.5
            
            # Penalizar desbalance entre depósitos
            balance_penalty = 50 * std_dev
        else:
            balance_penalty = 0
        
        # Fitness final (minimizar)
        fitness_value = total_distance + depot_penalty + vehicle_penalty + balance_penalty
        
        return fitness_value
    
    def solve(self, generations=100, verbose=True, early_stopping=None):
        """
        Ejecuta el algoritmo BRKGA
        
        Args:
            generations: Número máximo de generaciones
            verbose: Si True, muestra información durante la ejecución
            early_stopping: Número de generaciones sin mejora para detener (None = adaptativo)
                
        Returns:
            best_solution: Mejor solución encontrada
            best_fitness: Fitness de la mejor solución
            best_chromosome: Cromosoma correspondiente a la mejor solución
            fitness_history: Lista con el histórico de fitness
            convergence_gen: Generación en la que se alcanzó la mejor solución
        """
        # Inicialización
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        best_chromosome = None
        fitness_history = []
        no_improvement_count = 0
        convergence_gen = 0
        
        # Calcular early stopping adaptativo si no se proporciona un valor
        if early_stopping is None:
            num_customers = len(self.data['customers'])
            num_depots = len(self.data['depots'])
            early_stopping = calculate_early_stopping(num_customers, num_depots)
        
        if verbose:
            print(f"\nIniciando BRKGA con población={self.population_size}, refinamiento={'activado' if self.use_refinement else 'desactivado'}")
            print(f"Early stopping adaptativo: {early_stopping} generaciones sin mejora")
        
        for gen in range(generations):
            # Evaluar población
            fitness_values = np.array([self.fitness(ind) for ind in population])
            
            # Actualizar mejor solución
            current_best_idx = np.argmin(fitness_values)
            current_fitness = fitness_values[current_best_idx]
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution, _ = self.decode(population[current_best_idx])
                best_chromosome = population[current_best_idx].copy()
                convergence_gen = gen
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            fitness_history.append(best_fitness)
            
            if verbose and gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {best_fitness:.2f}, Sin mejora: {no_improvement_count}/{early_stopping}")
            
            # Criterio de parada temprana
            if no_improvement_count >= early_stopping:
                if verbose:
                    print(f"Parada temprana en generación {gen} - {no_improvement_count} generaciones sin mejora")
                break
            
            # Evolucionar población
            population = self.evolve(population, fitness_values)
        
        # Decodificar la mejor solución encontrada (sin penalización)
        best_solution, best_distance = self.decode(best_chromosome)
        
        if verbose:
            print("\nMejor solución encontrada:")
            used_depots = set()
            for i, route in enumerate(best_solution):
                print(f"Ruta {i+1} (Depósito {route['depot_id']}): {len(route['customers'])} clientes")
                print(f"  Distancia: {route['distance']:.2f}, Carga: {route['load']}")
                used_depots.add(route['depot_id'])
            print(f"Distancia total: {best_distance:.2f}")
            print(f"Depósitos utilizados: {len(used_depots)}/{self.num_depots}")
            print(f"Convergencia en generación: {convergence_gen}")
        
        return best_solution, best_distance, best_chromosome, fitness_history, convergence_gen
        
def main():
    # Configuración para todos los experimentos
    population_size = 100
    elite_percent = 0.1
    mutants_percent = 0.1
    max_generations = 200
    use_refinement = True  # Por defecto, activar el refinamiento
    
    # Verificar si el directorio de datos existe
    data_dir = '../dat'
    if not os.path.exists(data_dir):
        print(f"❌ El directorio {data_dir} no existe. Creándolo...")
        os.makedirs(data_dir)
        print(f"Por favor, coloca los archivos de instancias .dat en el directorio {data_dir} y vuelve a ejecutar.")
        return
    
    # Obtener todas las instancias disponibles
    instance_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    instance_files.sort()
    
    if not instance_files:
        print(f"❌ No se encontraron archivos .dat en {data_dir}")
        return
    
    print(f"Se encontraron {len(instance_files)} instancias en {data_dir}")
    
    # Verificar si hay resultados previos
    resultados_file = '../resultados.json'
    has_previous_results = os.path.exists(resultados_file)
    completed_instances = set()
    
    if has_previous_results:
        try:
            with open(resultados_file, 'r') as f:
                previous_results = json.load(f)
            for instance in previous_results:
                if 'error' not in previous_results[instance]:
                    completed_instances.add(instance)
            print(f"Se encontraron {len(completed_instances)} instancias ya procesadas.")
        except Exception as e:
            print(f"Error al cargar resultados previos: {str(e)}")
            has_previous_results = False
    
    # Menú para elegir modo de ejecución
    print("\n" + "="*60)
    print("BRKGA para MDVRP - Menú Principal")
    print("="*60)
    print("1. Ejecutar todas las instancias")
    print("2. Continuar con instancias pendientes" if has_previous_results else "2. [No disponible] Continuar con instancias pendientes")
    print("3. Ejecutar una instancia específica")
    print("4. Generar visualizaciones y resumen de resultados" if has_previous_results else "4. [No disponible] Generar visualizaciones")
    print("5. Analizar impacto del refinamiento")
    print("6. Analizar impacto del early stopping adaptativo" if has_previous_results else "6. [No disponible] Analizar early stopping")
    print("7. Configurar parámetros del algoritmo")
    print("8. Salir")
    
    choice = input("\nSeleccione una opción: ").strip()
    
    # Inicializar resultados
    results = {}
    
    if choice == '6' and has_previous_results:
        # Analizar impacto del early stopping adaptativo
        print("\nAnálisis del early stopping adaptativo")
        print("="*60)
        
        # Cargar resultados previos
        with open(resultados_file, 'r') as f:
            results = json.load(f)
        
        # Verificar si hay suficientes datos para analizar
        valid_results = {k: v for k, v in results.items() if 'error' not in v and 'early_stopping' in v}
        
        if not valid_results:
            print("❌ No hay resultados con información de early stopping para analizar.")
            print("   Ejecute algunas instancias primero y vuelva a intentarlo.")
            return main()
        
        print(f"Analizando datos de {len(valid_results)} instancias...")
        analyze_early_stopping(results)
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
    
    elif choice == '5':
        # Analizar impacto del refinamiento
        print("\nAnálisis de impacto del refinamiento")
        print("="*60)
        
        # Mostrar lista de instancias disponibles
        print("\nSeleccione una instancia para analizar:")
        for i, file in enumerate(instance_files):
            print(f"{i+1}. {file}")
        
        # Seleccionar una instancia
        while True:
            try:
                idx = int(input("\nSeleccione el número de la instancia: ").strip()) - 1
                if 0 <= idx < len(instance_files):
                    selected_instance = instance_files[idx]
                    break
                else:
                    print(f"❌ Número inválido. Debe estar entre 1 y {len(instance_files)}")
            except ValueError:
                print("❌ Por favor, ingrese un número válido")
        
        # Establecer parámetros para el análisis
        print("\nConfigurar parámetros para el análisis:")
        
        try:
            pop_size = input(f"Tamaño de población [{population_size}]: ").strip()
            if pop_size:
                population_size = int(pop_size)
            
            elite = input(f"Porcentaje de élite [{elite_percent}]: ").strip()
            if elite:
                elite_percent = float(elite)
            
            mutants = input(f"Porcentaje de mutantes [{mutants_percent}]: ").strip()
            if mutants:
                mutants_percent = float(mutants)
            
            generations = input(f"Número máximo de generaciones [200]: ").strip()
            if generations:
                max_generations_analysis = int(generations)
            else:
                max_generations_analysis = 200
            
            # Ejecutar análisis
            print(f"\nIniciando análisis comparativo para {selected_instance}...")
            analyze_refinement_impact(
                data_dir, 
                selected_instance, 
                population_size, 
                elite_percent, 
                mutants_percent, 
                max_generations_analysis
            )
            
        except ValueError as e:
            print(f"❌ Error en la entrada: {str(e)}")
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
        
    elif choice == '7':
        # Configurar parámetros
        print("\n" + "="*60)
        print("Configuración de Parámetros")
        print("="*60)
        
        try:
            pop_size = input(f"Tamaño de población [{population_size}]: ").strip()
            if pop_size:
                population_size = int(pop_size)
            
            elite = input(f"Porcentaje de élite [{elite_percent}]: ").strip()
            if elite:
                elite_percent = float(elite)
            
            mutants = input(f"Porcentaje de mutantes [{mutants_percent}]: ").strip()
            if mutants:
                mutants_percent = float(mutants)
            
            generations = input(f"Número máximo de generaciones [{max_generations}]: ").strip()
            if generations:
                max_generations = int(generations)
            
            refine = input(f"Activar refinamiento (s/n) [{'s' if use_refinement else 'n'}]: ").strip().lower()
            if refine:
                use_refinement = refine == 's'
            
            # Configuración de early stopping adaptativo
            print("\nConfiguración de early stopping adaptativo:")
            base_factor = input("Factor base para early stopping [1000]: ").strip()
            if base_factor:
                # Modificar la función calculate_early_stopping con el nuevo factor base
                calculate_early_stopping.__defaults__ = (int(base_factor), 20, 200)
                print(f"Factor base para early stopping actualizado a: {int(base_factor)}")
            
            print("\nConfiguración actualizada:")
            print(f"- Tamaño de población: {population_size}")
            print(f"- Porcentaje de élite: {elite_percent}")
            print(f"- Porcentaje de mutantes: {mutants_percent}")
            print(f"- Máximo de generaciones: {max_generations}")
            print(f"- Refinamiento: {'Activado' if use_refinement else 'Desactivado'}")
            print(f"- Factor base para early stopping: {calculate_early_stopping.__defaults__[0]}")
            
            # Volver al menú principal
            input("\nPresione Enter para volver al menú principal...")
            return main()
            
        except ValueError as e:
            print(f"❌ Error en la entrada: {str(e)}")
            return main()
    
    elif choice == '1':
        # Ejecutar todas las instancias (ignorando resultados previos)
        if has_previous_results:
            confirm = input("\n⚠️ Ya existen resultados previos. ¿Desea sobreescribirlos? (s/n): ").strip().lower()
            if confirm != 's':
                print("Operación cancelada.")
                return
        
        run_all_instances(data_dir, instance_files, population_size, elite_percent, mutants_percent, max_generations, results, use_refinement)
        
        # Generar tabla de resumen y visualizaciones para todas las instancias
        print("\nGenerando tabla de resumen y visualizaciones...")
        generate_summary_table(results)
        generate_visualizations(results)
        print("\n✅ Proceso completo. Resultados guardados en 'resultados.json'")
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
        
    elif choice == '2' and has_previous_results:
        # Cargar resultados previos
        with open(resultados_file, 'r') as f:
            results = json.load(f)
        
        # Preguntar si quiere mantener la configuración de refinamiento anterior
        prev_refinement = None
        for _, data in results.items():
            if 'use_refinement' in data:
                prev_refinement = data['use_refinement']
                break
        
        if prev_refinement is not None and prev_refinement != use_refinement:
            print(f"\nAviso: Las ejecuciones anteriores usaron refinamiento: {'Activado' if prev_refinement else 'Desactivado'}")
            print(f"La configuración actual es: {'Activado' if use_refinement else 'Desactivado'}")
            refine_choice = input("¿Desea mantener la configuración anterior? (s/n): ").strip().lower()
            if refine_choice == 's':
                use_refinement = prev_refinement
                print(f"Usando refinamiento: {'Activado' if use_refinement else 'Desactivado'}")
        
        # Continuar con instancias pendientes
        run_all_instances(data_dir, instance_files, population_size, elite_percent, mutants_percent, max_generations, results, use_refinement)
        
        # Generar tabla de resumen y visualizaciones
        print("\nGenerando tabla de resumen y visualizaciones...")
        generate_summary_table(results)
        generate_visualizations(results)
        print("\n✅ Proceso completo. Resultados guardados en 'resultados.json'")
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
        
    elif choice == '3':
        # Mostrar lista de instancias disponibles
        print("\nInstancias disponibles:")
        
        # Mostrar cuáles instancias ya están procesadas
        for i, file in enumerate(instance_files):
            status = "✅ [Procesada]" if file in completed_instances else "⏳ [Pendiente]"
            print(f"{i+1}. {file} {status}")
        
        # Seleccionar una instancia
        while True:
            try:
                idx = int(input("\nSeleccione el número de la instancia a ejecutar: ").strip()) - 1
                if 0 <= idx < len(instance_files):
                    selected_instance = instance_files[idx]
                    break
                else:
                    print(f"❌ Número inválido. Debe estar entre 1 y {len(instance_files)}")
            except ValueError:
                print("❌ Por favor, ingrese un número válido")
        
        # Confirmar si ya está procesada
        if selected_instance in completed_instances:
            confirm = input(f"\n⚠️ La instancia {selected_instance} ya ha sido procesada. ¿Desea procesarla nuevamente? (s/n): ").strip().lower()
            if confirm != 's':
                # Cargar resultados existentes para visualizar
                print("\nCargando resultados existentes para visualización...")
                with open(resultados_file, 'r') as f:
                    prev_results = json.load(f)
                
                # Cargar datos para visualización
                mdvrp_data = parse_mdvrp_file(os.path.join(data_dir, selected_instance))
                
                instance_result = prev_results[selected_instance]
                instance_result['data'] = mdvrp_data
                
                visualize_instance_solution(instance_result, data_dir)
                input("\nPresione Enter para volver al menú principal...")
                return main()
        
        # Preguntar si desea ejecutar con refinamiento
        refine_choice = input(f"\n¿Desea utilizar refinamiento para esta instancia? (s/n) [{'s' if use_refinement else 'n'}]: ").strip().lower()
        if refine_choice:
            use_refinement = refine_choice == 's'
        
        # Ejecutar solo la instancia seleccionada
        instance_results = run_single_instance(
            data_dir, selected_instance, 
            population_size, elite_percent, mutants_percent, max_generations, use_refinement
        )
        
        if instance_results and 'error' not in instance_results:
            # Visualizar la instancia y su solución
            visualize_instance_solution(instance_results, data_dir)
            
        # Actualizar el archivo de resultados con esta instancia
        if has_previous_results:
            with open(resultados_file, 'r') as f:
                all_results = json.load(f)
            
            # Agregar/actualizar esta instancia
            if instance_results:
                instance_name = instance_results['instance']
                serializable_result = {k: v for k, v in instance_results.items() if k != 'data'}
                all_results[instance_name] = serializable_result
                
                with open(resultados_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"\n✅ Resultados actualizados en {resultados_file}")
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
        
    elif choice == '4' and has_previous_results:
        # Solo generar visualizaciones a partir de resultados existentes
        with open(resultados_file, 'r') as f:
            results = json.load(f)
        
        print("\nGenerando tabla de resumen y visualizaciones...")
        generate_summary_table(results)
        generate_visualizations(results)
        
        # Preguntar si quiere analizar también el early stopping
        if any('early_stopping' in data for _, data in results.items() if 'error' not in data):
            analyze_es = input("\n¿Desea analizar también el early stopping adaptativo? (s/n): ").strip().lower()
            if analyze_es == 's':
                analyze_early_stopping(results)
        
        print("\n✅ Visualizaciones generadas a partir de 'resultados.json'")
        
        input("\nPresione Enter para volver al menú principal...")
        return main()
        
    elif choice == '8':
        print("\nSaliendo del programa...")
        return
    
    else:
        print("\n❌ Opción inválida o no disponible.")
        input("\nPresione Enter para volver al menú principal...")
        return main()

if __name__ == "__main__":
    main()