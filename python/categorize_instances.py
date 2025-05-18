import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
from BRKGA import parse_mdvrp_file


def select_benchmark_instances(representatives, include_extremes=True, num_extra_per_category=1):
    """
    Selecciona instancias representativas para benchmarking de hiperparámetros.

    Args:
        representatives: Diccionario con información de categorías y sus instancias
        include_extremes: Si True, incluye la instancia más simple y más compleja
        num_extra_per_category: Número de instancias adicionales por categoría

    Returns:
        list: Lista de instancias seleccionadas para el benchmark
    """
    benchmark_instances = []
    benchmark_by_category = {}

    for category, data in representatives.items():
        category_benchmarks = {
            'representative': data['representative'],
            'additional': []
        }

        # Siempre añadir la instancia representativa (central del cluster)
        benchmark_instances.append(data['representative'])

        # Opcionalmente añadir las instancias extremas (más simple y más compleja)
        if include_extremes:
            # Añadir sólo si son diferentes de la representativa
            if data['simplest'] != data['representative']:
                benchmark_instances.append(data['simplest'])
                category_benchmarks['simplest'] = data['simplest']

            if data['most_complex'] != data['representative'] and data['most_complex'] != data['simplest']:
                benchmark_instances.append(data['most_complex'])
                category_benchmarks['most_complex'] = data['most_complex']

        # Si se solicitan instancias adicionales
        if num_extra_per_category > 0 and len(data['instances']) > 1:
            # Excluir las instancias que ya hemos añadido
            already_added = [data['representative']]
            if include_extremes:
                already_added.extend([data['simplest'], data['most_complex']])

            remaining = [inst for inst in data['instances']
                         if inst not in already_added]

            if remaining:
                # Estrategia de selección: instancias uniformemente distribuidas
                if len(remaining) <= num_extra_per_category:
                    additional = remaining
                else:
                    # Seleccionar instancias con espaciado uniforme
                    step = len(remaining) // num_extra_per_category
                    indices = range(0, len(remaining), step)[:num_extra_per_category]
                    additional = [remaining[i] for i in indices]

                benchmark_instances.extend(additional)
                category_benchmarks['additional'] = additional

        benchmark_by_category[category] = category_benchmarks

    # Eliminar duplicados manteniendo el orden
    benchmark_instances = list(dict.fromkeys(benchmark_instances))

    print("\n" + "=" * 70)
    print("INSTANCIAS SELECCIONADAS PARA BENCHMARKING DE HIPERPARÁMETROS")
    print("=" * 70)

    print("\nPor categoría:")
    for category, data in benchmark_by_category.items():
        print(f"\nCategoría {category + 1}:")
        print(f"  - Representativa: {data['representative']}")

        if 'simplest' in data:
            print(f"  - Más simple: {data['simplest']}")

        if 'most_complex' in data:
            print(f"  - Más compleja: {data['most_complex']}")

        if data['additional']:
            print(f"  - Adicionales: {', '.join(data['additional'])}")

    print("\nListado completo:")
    for i, instance in enumerate(benchmark_instances):
        print(f"{i + 1}. {instance}")

    # Guardar la lista en un archivo
    with open('../instances_selection/benchmark_instances.txt', 'w') as f:
        f.write("INSTANCIAS SELECCIONADAS PARA BENCHMARKING\n")
        f.write("=" * 50 + "\n\n")

        f.write("INSTANCIAS REPRESENTATIVAS POR CATEGORÍA\n")
        f.write("=" * 50 + "\n\n")

        for category, data in benchmark_by_category.items():
            f.write(f"Categoría {category + 1}: {data['representative']}\n")

        f.write("\n\n")
        f.write("=" * 50 + "\n")
        f.write("INSTANCIAS POR CATEGORÍA\n")
        f.write("=" * 50 + "\n\n")

        for category, data in benchmark_by_category.items():
            f.write(f"Categoría {category + 1}:\n")
            for inst in representatives[category]['instances']:
                f.write(f"  - {inst}\n")
            f.write("\n")

    return benchmark_instances


def categorize_instances(data_dir, num_categories=3, visualize=True):
    """
    Categoriza las instancias MDVRP disponibles según múltiples criterios
    para identificar instancias representativas de cada categoría.

    Args:
        data_dir: Directorio con los archivos de instancias
        num_categories: Número de categorías a identificar
        visualize: Si True, genera visualizaciones de las categorías

    Returns:
        dict: Diccionario con las categorías y sus instancias representativas
    """
    # 1. Cargar y analizar las instancias
    print(f"Analizando instancias en {data_dir}...")
    instance_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]

    if not instance_files:
        print(f"No se encontraron archivos .dat en {data_dir}")
        return {}

    print(f"Se encontraron {len(instance_files)} instancias")

    # 2. Extraer características de cada instancia
    instance_data = []

    for instance_file in instance_files:
        file_path = os.path.join(data_dir, instance_file)
        try:
            # Parsear la instancia
            mdvrp_data = parse_mdvrp_file(file_path)

            # Características básicas
            num_customers = mdvrp_data['num_customers']
            num_depots = mdvrp_data['num_depots']
            num_vehicles = mdvrp_data['num_vehicles']

            # Calcular características derivadas
            total_demand = sum(c['demand'] for c in mdvrp_data['customers'])
            avg_demand = total_demand / num_customers if num_customers > 0 else 0
            max_demand = max(c['demand'] for c in mdvrp_data['customers']) if mdvrp_data['customers'] else 0

            # Características espaciales
            all_x = [d['x'] for d in mdvrp_data['depots']] + [c['x'] for c in mdvrp_data['customers']]
            all_y = [d['y'] for d in mdvrp_data['depots']] + [c['y'] for c in mdvrp_data['customers']]

            area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))
            density = (num_customers + num_depots) / area if area > 0 else 0

            # Calcular dispersión espacial (distancia media entre puntos)
            points = [(c['x'], c['y']) for c in mdvrp_data['customers']] + [(d['x'], d['y']) for d in
                                                                            mdvrp_data['depots']]

            avg_distance = 0
            if len(points) > 1:
                sum_dist = 0
                count = 0
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        dist = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                        sum_dist += dist
                        count += 1
                avg_distance = sum_dist / count if count > 0 else 0

            # Calcular relación entre clientes y vehículos
            clients_per_vehicle = num_customers / num_vehicles if num_vehicles > 0 else 0

            # Ratio de capacidad total vs demanda total
            max_capacity = mdvrp_data['vehicle_info'][0]['max_load'] * num_vehicles
            capacity_demand_ratio = max_capacity / total_demand if total_demand > 0 else 0

            # Guardar características en una lista
            instance_data.append({
                'instance': instance_file,
                'num_customers': num_customers,
                'num_depots': num_depots,
                'num_vehicles': num_vehicles,
                'total_demand': total_demand,
                'avg_demand': avg_demand,
                'max_demand': max_demand,
                'area': area,
                'density': density,
                'avg_distance': avg_distance,
                'clients_per_vehicle': clients_per_vehicle,
                'capacity_demand_ratio': capacity_demand_ratio
            })

            print(f"✅ {instance_file}: {num_customers} clientes, {num_depots} depósitos, {num_vehicles} vehículos")

        except Exception as e:
            print(f"❌ Error analizando {instance_file}: {str(e)}")

    if not instance_data:
        print("No se pudieron analizar instancias")
        return {}

    # 3. Convertir a DataFrame para análisis
    df = pd.DataFrame(instance_data)

    # 4. Visualizar distribuciones de características
    if visualize:
        plt.figure(figsize=(15, 10))

        # Variables a visualizar
        vars_to_plot = ['num_customers', 'num_depots', 'num_vehicles',
                        'avg_demand', 'density', 'avg_distance',
                        'capacity_demand_ratio']

        for i, var in enumerate(vars_to_plot):
            plt.subplot(3, 3, i + 1)
            sns.histplot(df[var], kde=True)
            plt.title(f'Distribución de {var}')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('distribucion_caracteristicas.png', dpi=300)
        plt.show()

    # 5. Aplicar K-means para categorizar instancias
    # Seleccionar características numéricas para clustering
    features = ['num_customers', 'num_depots', 'num_vehicles',
                'avg_demand', 'density', 'avg_distance',
                'capacity_demand_ratio']

    X = df[features].values

    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determinar número óptimo de clusters si no se especifica
    if num_categories is None:
        wcss = []
        for i in range(1, min(10, len(X_scaled))):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # Visualizar el método del codo
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, min(10, len(X_scaled))), wcss, marker='o')
            plt.title('Método del Codo para K-Means')
            plt.xlabel('Número de Clusters')
            plt.ylabel('WCSS')
            plt.grid(True, alpha=0.3)
            plt.savefig('metodo_codo.png', dpi=300)
            plt.show()

        # Encontrar el codo de la curva (punto de inflexión)
        deltas = np.diff(wcss)
        k_optimal = np.argmax(np.diff(deltas)) + 2
        num_categories = min(k_optimal, 5)  # Limitar a máximo 5 categorías

    # Aplicar K-means con el número de categorías determinado
    kmeans = KMeans(n_clusters=num_categories, random_state=42, n_init=10)
    df['category'] = kmeans.fit_predict(X_scaled)

    # 6. Visualizar las categorías
    if visualize:
        # Reducir dimensionalidad para visualización (usando PCA)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Añadir componentes principales al DataFrame
        df['PC1'] = X_pca[:, 0]
        df['PC2'] = X_pca[:, 1]

        # Visualizar clusters en espacio 2D
        plt.figure(figsize=(12, 8))

        sns.scatterplot(x='PC1', y='PC2', hue='category',
                        palette='viridis', s=100, data=df)

        # Añadir nombres de instancias como anotaciones
        for i, row in df.iterrows():
            plt.annotate(row['instance'],
                         xy=(row['PC1'], row['PC2']),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=8, alpha=0.7)

        plt.title('Categorización de Instancias MDVRP')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('categorias_instancias.png', dpi=300)
        plt.show()

        # Visualizar características promedio por categoría
        category_means = df.groupby('category')[features].mean()

        plt.figure(figsize=(14, 10))
        sns.heatmap(category_means, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('Características Promedio por Categoría')
        plt.savefig('caracteristicas_por_categoria.png', dpi=300)
        plt.show()

    # 7. Identificar instancias representativas de cada categoría
    representatives = {}

    for category in range(num_categories):
        category_instances = df[df['category'] == category]

        if len(category_instances) == 0:
            continue

        # Calcular el centroide de la categoría
        centroid = kmeans.cluster_centers_[category]

        # Encontrar la instancia más cercana al centroide
        category_X = X_scaled[df['category'] == category]
        distances = np.linalg.norm(category_X - centroid, axis=1)
        closest_idx = np.argmin(distances)

        representative_instance = category_instances.iloc[closest_idx]['instance']

        # Identificar también las instancias más simples y más complejas de la categoría
        complexity_score = category_instances['num_customers'] * 2 + category_instances['num_depots'] + \
                           category_instances['density'] * 10 + category_instances['avg_distance']

        simplest_instance = category_instances.loc[complexity_score.idxmin()]['instance']
        most_complex_instance = category_instances.loc[complexity_score.idxmax()]['instance']

        # Guardar información de la categoría
        representatives[category] = {
            'representative': representative_instance,
            'simplest': simplest_instance,
            'most_complex': most_complex_instance,
            'size': len(category_instances),
            'instances': category_instances['instance'].tolist(),
            'characteristics': {
                feature: category_means.loc[category, feature] for feature in features
            },
            'complexity_range': (complexity_score.min(), complexity_score.max())
        }

    # 8. Mostrar resumen de categorización
    print("\n" + "=" * 80)
    print("CATEGORIZACIÓN DE INSTANCIAS MDVRP")
    print("=" * 80)

    for category, data in representatives.items():
        print(f"\nCategoría {category + 1}:")
        print(f"  Tamaño: {data['size']} instancias")
        print(f"  Instancia representativa: {data['representative']}")
        print("  Características promedio:")
        for feature, value in data['characteristics'].items():
            print(f"    - {feature}: {value:.2f}")

        # Mostrar algunas instancias de la categoría (máximo 5)
        instances_to_show = data['instances'][:5]
        print(f"  Algunas instancias: {', '.join(instances_to_show)}")
        if len(data['instances']) > 5:
            print(f"    ... y {len(data['instances']) - 5} más")

    # 9. Guardar categorización en un archivo CSV
    df.to_csv('categorias_instancias.csv', index=False)
    print("\n✅ Categorización guardada en 'categorias_instancias.csv'")

    # Guardar instancias representativas en un archivo
    with open('../instances_selection/instancias_representativas.txt', 'w') as f:
        f.write("INSTANCIAS REPRESENTATIVAS POR CATEGORÍA\n")
        f.write("=" * 50 + "\n\n")
        for category, data in representatives.items():
            f.write(f"Categoría {category + 1}: {data['representative']}\n")

        f.write("\n\n" + "=" * 50 + "\n")
        f.write("INSTANCIAS POR CATEGORÍA\n")
        f.write("=" * 50 + "\n\n")
        for category, data in representatives.items():
            f.write(f"Categoría {category + 1}:\n")
            for instance in data['instances']:
                f.write(f"  - {instance}\n")
            f.write("\n")

    print("✅ Instancias representativas guardadas en 'instancias_representativas.txt'")

    return representatives


def select_benchmark_instances(representatives, include_extremes=True, num_extra_per_category=0):
    """
    Selecciona instancias de referencia para benchmarking considerando
    las categorías identificadas, priorizando instancias representativas
    y los casos extremos (más simple y más complejo).

    Args:
        representatives: Resultado de categorize_instances()
        include_extremes: Si True, incluye la instancia más simple y más compleja de cada categoría
        num_extra_per_category: Número de instancias adicionales aleatorias por categoría

    Returns:
        dict: Diccionario con tipos de instancias seleccionadas por categoría
        list: Lista de todas las instancias seleccionadas para benchmarking
    """
    benchmark_instances = []
    benchmark_by_category = {}

    for category, data in representatives.items():
        category_benchmarks = {
            'representative': data['representative'],
            'additional': []
        }

        # Siempre añadir la instancia representativa
        benchmark_instances.append(data['representative'])

        # Opcionalmente añadir las instancias extremas (más simple y más compleja)
        if include_extremes:
            # Añadir sólo si son diferentes de la representativa
            if data['simplest'] != data['representative']:
                benchmark_instances.append(data['simplest'])
                category_benchmarks['simplest'] = data['simplest']

            if data['most_complex'] != data['representative'] and data['most_complex'] != data['simplest']:
                benchmark_instances.append(data['most_complex'])
                category_benchmarks['most_complex'] = data['most_complex']

        # Si se solicitan instancias adicionales y hay suficientes
        if num_extra_per_category > 0 and len(data['instances']) > 1:
            # Excluir las instancias que ya hemos añadido
            already_added = [data['representative']]
            if include_extremes:
                already_added.extend([data['simplest'], data['most_complex']])

            remaining = [inst for inst in data['instances']
                         if inst not in already_added]

            if remaining:
                # Tomar algunas instancias adicionales distribuidas uniformemente en complejidad
                if len(remaining) <= num_extra_per_category:
                    additional = remaining
                else:
                    # Estrategia de selección: tomar instancias uniformemente distribuidas
                    # en el rango de complejidad para tener buena cobertura
                    step = len(remaining) // num_extra_per_category
                    indices = range(0, len(remaining), step)[:num_extra_per_category]
                    additional = [remaining[i] for i in indices]

                benchmark_instances.extend(additional)
                category_benchmarks['additional'] = additional

        benchmark_by_category[category] = category_benchmarks

    # Eliminar duplicados manteniendo el orden
    benchmark_instances = list(dict.fromkeys(benchmark_instances))

    print("\n" + "=" * 70)
    print("INSTANCIAS SELECCIONADAS PARA BENCHMARKING DE HIPERPARÁMETROS")
    print("=" * 70)

    print("\nPor categoría:")
    for category, data in benchmark_by_category.items():
        print(f"\nCategoría {category + 1}:")
        print(f"  - Representativa: {data['representative']}")

        if 'simplest' in data:
            print(f"  - Más simple: {data['simplest']}")

        if 'most_complex' in data:
            print(f"  - Más compleja: {data['most_complex']}")

        if data['additional']:
            print(f"  - Adicionales: {', '.join(data['additional'])}")

    print("\nListado completo:")
    for i, instance in enumerate(benchmark_instances):
        print(f"{i + 1}. {instance}")

    # Guardar selección en un archivo
    with open('../instances_selection/benchmark_instances.txt', 'w') as f:
        f.write("INSTANCIAS SELECCIONADAS PARA BENCHMARKING\n\n")

        for i, instance in enumerate(benchmark_instances):
            f.write(f"{i + 1}. {instance}\n")

    print("\n✅ Lista de instancias guardada en 'benchmark_instances.txt'")

    return benchmark_by_category, benchmark_instances


# Función para integrar en el menú principal de BRKGA.py
def menu_categorize_instances():
    """
    Función para el menú que permite categorizar instancias y
    seleccionar un conjunto representativo para benchmarking
    """
    data_dir = '../dat'

    print("\n" + "=" * 70)
    print("CATEGORIZACIÓN DE INSTANCIAS MDVRP PARA BENCHMARKING")
    print("=" * 70)

    # Verificar directorio de datos
    if not os.path.exists(data_dir):
        print(f"❌ El directorio {data_dir} no existe.")
        return

    # Opciones de categorización
    print("\nOpciones de categorización:")
    print("1. Determinar automáticamente el número óptimo de categorías")
    print("2. Especificar manualmente el número de categorías")

    choice = input("\nSeleccione una opción: ").strip()

    # Número de categorías
    if choice == '1':
        num_categories = None  # Automático
        print("Se determinará automáticamente el número óptimo de categorías.")
    elif choice == '2':
        try:
            num_categories = int(input("Ingrese el número de categorías deseado (2-8): ").strip())
            if num_categories < 2 or num_categories > 8:
                print("⚠️ Número inválido. Se usará el valor predeterminado de 4 categorías.")
                num_categories = 4
        except ValueError:
            print("⚠️ Entrada inválida. Se usará el valor predeterminado de 4 categorías.")
            num_categories = 4
    else:
        print("⚠️ Opción inválida. Se determinará automáticamente el número de categorías.")
        num_categories = None

    # Confirmar visualizaciones
    visualize = input("\n¿Desea generar visualizaciones? (s/n): ").strip().lower() == 's'

    # Ejecutar categorización
    print("\nAnalizando y categorizando instancias...")
    representatives = categorize_instances(data_dir, num_categories=num_categories, visualize=visualize)

    if not representatives:
        print("❌ No se pudo completar la categorización.")
        return

    # Opciones para selección de instancias de benchmarking
    print("\nSelección de instancias para benchmarking:")
    print("1. Solo instancias representativas (1 por categoría)")
    print("2. Instancias representativas + casos extremos (simple y complejo)")
    print("3. Representativas + extremos + adicionales")

    selection = input("\nSeleccione una opción: ").strip()

    include_extremes = selection in ['2', '3']

    num_extra = 0
    if selection == '3':
        try:
            num_extra = int(input("Instancias adicionales por categoría: ").strip())
        except ValueError:
            print("⚠️ Valor inválido. Se usará 1 instancia adicional por categoría.")
            num_extra = 1

    # Seleccionar instancias para benchmarking
    benchmark_by_category, benchmark_instances = select_benchmark_instances(
        representatives,
        include_extremes=include_extremes,
        num_extra_per_category=num_extra
    )

    # Preguntar si quiere ejecutar un benchmarking de hiperparámetros
    run_benchmark = input(
        "\n¿Desea ejecutar un análisis de hiperparámetros con estas instancias? (s/n): ").strip().lower() == 's'

    if run_benchmark:
        # Aquí se podría añadir la integración con la función de benchmarking de hiperparámetros
        print("\n⚠️ La función de benchmarking de hiperparámetros no está implementada aún.")
        print("   Puede usar la lista de instancias generada en 'benchmark_instances.txt'")

    return benchmark_instances


# Ejemplo de uso:
if __name__ == "__main__":
    # Obtener el directorio de datos
    data_dir = '../dat'

    # Verificar si el directorio existe
    if not os.path.exists(data_dir):
        print(f"❌ El directorio {data_dir} no existe. Creándolo...")
        os.makedirs(data_dir)
        print(f"Por favor, coloca los archivos de instancias .dat en el directorio {data_dir} y vuelve a ejecutar.")
    else:
        # Categorizar instancias
        print("Categorizando instancias...")
        representatives = categorize_instances(data_dir, num_categories=3, visualize=True)

        if representatives:
            # Seleccionar instancias para benchmarking
            benchmark_instances = select_benchmark_instances(
                representatives,
                include_extremes=True,
                num_extra_per_category=1
            )

            print(f"\nSe seleccionaron {len(benchmark_instances)} instancias para benchmark.")
            print("Puede utilizar estas instancias para probar diferentes configuraciones de hiperparámetros.")
        else:
            print("❌ No se pudieron categorizar las instancias.")