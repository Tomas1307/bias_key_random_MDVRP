# BRKGA para el Problema de Rutas de Vehículos con Múltiples Depósitos (MDVRP)

## Descripción General del Proyecto

Este repositorio contiene una implementación en C++ de un Algoritmo Genético con Claves Aleatorias Sesgadas (BRKGA) para resolver el Problema de Rutas de Vehículos con Múltiples Depósitos (MDVRP). El proyecto incluye código para:

1. La implementación principal del algoritmo BRKGA
2. Pruebas de hiperparámetros para la optimización del algoritmo
3. Herramientas de análisis y visualización
4. Instancias de referencia y resultados

## Definición del Problema

El Problema de Rutas de Vehículos con Múltiples Depósitos (MDVRP) es una extensión del clásico Problema de Rutas de Vehículos (VRP) donde se consideran múltiples depósitos. El objetivo es determinar el conjunto óptimo de rutas para una flota de vehículos que deben atender a un conjunto dado de clientes, minimizando la distancia total recorrida, y sujeto a las siguientes restricciones:

- Cada cliente debe ser visitado exactamente una vez
- Cada vehículo debe comenzar y terminar en el mismo depósito
- Se deben respetar las restricciones de capacidad de los vehículos
- Se deben respetar las restricciones de duración de las rutas

## Estructura del Repositorio

```
MDVRP/
├── dat/                            # Instancias de referencia
├── hp_results/                     # Resultados de pruebas de hiperparámetros
│   ├── hyperparameter_results.json
│   └── hyperparameter_results/
│       ├── advanced_plots/
│       ├── plots/
│       ├── summary_report/
│       ├── best_configuration.json
│       ├── hyperparameter_results.csv
│       └── p01.txt, p02.txt, ...   # Resultados para cada instancia
├── instances_selection/            # Selección y categorización de instancias
│   ├── benchmark_instances.txt
│   ├── caracteristicas_por_categoria.png
│   ├── categorias_instancias.csv
│   ├── distribucion_caracteristicas.png
│   ├── instancias_representativas.txt
│   └── metodo_codo.png
├── python/                         # Scripts de análisis en Python
│   ├── plots/
│   │   ├── clasificacion_gaps.png
│   │   ├── comparacion_fitness_bks.png
│   │   ├── convergencia_*.png      # Gráficos de convergencia por instancia
│   │   ├── mejora_acumulada.png
│   │   └── tiempo_vs_complejidad.png
│   ├── BRKGA.py
│   ├── categorize_instances.py
│   ├── hyperparameter_results.py
│   ├── results_plot.py
│   └── results.py
└── results_cpp/                    # Resultados de ejecución en C++
    ├── p01.txt, p02.txt, ...       # Resultados para instancias individuales
    ├── brkga_mdvrp.cpp             # Implementación principal de BRKGA
    ├── brkga.exe
    ├── compile.bat
    ├── DESCRIPTION.md              # Descripción detallada del formato de archivos
    ├── hyperparameter_test.cpp     # Código para pruebas de hiperparámetros
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── resultados.json
    ├── resumen_resultados.csv
    └── table_results.ipynb
```

## Implementación del Algoritmo

La implementación utiliza un enfoque de Algoritmo Genético con Claves Aleatorias Sesgadas (BRKGA), que es una variante de los algoritmos genéticos que:

- Utiliza una representación cromosómica con números reales en el rango [0,1]
- Mantiene poblaciones separadas de elite y no-elite
- Utiliza un cruce sesgado que favorece genes de padres elite
- Aplica un decodificador específico para el problema que convierte los cromosomas en soluciones

### Características Principales

- Gestión adaptativa de la población
- Refinamiento de búsqueda local 2-opt
- Criterios de parada temprana basados en el tamaño del problema
- Codificación para asignación de depósitos y secuenciación de clientes
- Función de aptitud basada en penalizaciones

## Parámetros Principales

El algoritmo utiliza los siguientes parámetros clave:

```cpp
const int POPULATION_SIZE = 250;    
const double ELITE_PERCENT = 0.1;
const double MUTANTS_PERCENT = 0.2;
const int MAX_GENERATIONS = 250;
const double P_BIAS = 0.8;
const bool USE_REFINEMENT = true;
```

## Uso

### Compilando el Código

Utilice el archivo `compile.bat` incluido o el `Makefile`:

```bash
# Usando make
make

# O usando el archivo batch en Windows
compile.bat
```

### Ejecutando el Algoritmo Principal

```bash
# Procesar todas las instancias no procesadas en el directorio dat/
./brkga.exe

# Procesar una instancia específica
./brkga.exe p01

# O en Windows
.\brkga.exe pr04

# Puedes ejecutar múltiples instancias específicas
.\brkga.exe p01
.\brkga.exe p18
```

Esto te permite procesar todas las instancias o enfocarte en problemas de referencia específicos.

### Ejecutando Pruebas de Hiperparámetros

```bash
./hyperparameter_test.exe <nombre_instancia>
```

Ejemplo:
```bash
./hyperparameter_test.exe p01
```

## Instancias de Referencia

El repositorio incluye 33 instancias estándar de referencia para MDVRP:
- Instancias 1-7 creadas por Christofides y Eilon (1969)
- Instancias 8-11 descritas por Gillett y Johnson (1976)
- Instancias 12-23 propuestas por Chao et al. (1993)
- Instancias 24-33 propuestas por Cordeau et al. (1997)

Las descripciones detalladas de las instancias se pueden encontrar en `DESCRIPTION.md` y en el [sitio web del Grupo de Investigación NEO](http://neo.lcc.uma.es/vrp/vrp-instances/description-for-files-of-cordeaus-instances/).

## Análisis de Resultados

Los resultados de ejecución se almacenan en:
- `results_cpp/results.txt` - Resultados principales de ejecución
- `hp_results/hyperparameter_results/` - Resultados de pruebas de hiperparámetros

Se proporcionan scripts de Python para analizar y visualizar resultados:
- `results.py` - Procesa resultados
- `results_plot.py` - Crea visualizaciones
- `hyperparameter_results.py` - Analiza el impacto de los hiperparámetros

# Descricion para instancias de Cordeau

> Reference note: This description can be found in [NEO Research Group](http://neo.lcc.uma.es/vrp/vrp-instances/description-for-files-of-cordeaus-instances/) and it is fully reported below.
> [Return](https://github.com/fboliveira/MDVRP-Instances)

## Referencias

[1] Christofides, N., Eilon, S.: An algorithm for the vehicle-dispatching problem. Oper. Res. Q. 20(3), 309–318 (1969).

[2] Gillett, B., Johnson, J.: Multi-terminal vehicle-dispatch algorithm. Omega 4(6), 711–718 (1976).

[3] Chao, I., Golden, B., Wasil, E.: A new heuristic for the multi-depot vehicle routing problem that improves upon best-known solutions. Am. J. Math. Manag.Sci. 13(3), 371–406 (1993).

[4] Cordeau, J., Gendreau, M., Laporte, G.: A tabu search heuristic for periodic and multi-depot vehicle routing problems. Networks 30(2), 105–119 (1997).

[5] Cordeau, J., Maischberger, M.: A parallel iterated tabu search heuristic for vehicle routing problems. Comput. Oper. Res. 39(9), 2033–2050 (2012)

[6] Subramanian, A., Uchoa, E., Ochi, L.S.: A hybrid algorithm for a class of vehicle routing problems. Comput. Oper. Res. 40(10), 2519–2531 (2013)

[7] Vidal, T., Crainic, T., Gendreau, M., Lahrichi, N., Rei, W.: A hybrid genetic algorithm for multi-depot and periodic vehicle routing problems. Oper. Res. 60(3), 611–624 (2012).

[8] Escobar, J. W., Linfati, R., Toth, P., & Baldoquin, M. G. (2014). A hybrid granular tabu search algorithm for the multi-depot vehicle routing problem. Journal of Heuristics, 20(5), 483-509.

## Licencia

Este proyecto está licenciado bajo los términos incluidos en el archivo LICENSE.

## Autor

Tomas Acosta Bernal

t.acosta@uniandes.edu.co
