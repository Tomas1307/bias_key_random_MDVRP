# Analisis de Hiperparametros: Informe Resumido

Generado: 2025-04-07 10:13:53

## Informacion General

- Total de configuraciones evaluadas: 19206
- Instancias analizadas: 11
- Hiperparametros optimizados: 5

## Mejores Configuraciones por Instancia

| Instancia | Fitness | population_size | elite_percent | mutants_percent | max_generations | p_bias | Tiempo (s) | Generacion Convergencia |
|---|---|---|---|---|---|---|---|---|
| p01 | 687.24 | 250 | 0.1 | 0.3 | 200 | 0.7 | 42.26 | 47 |
| p02 | 534.50 | 250 | 0.1 | 0.4 | 200 | 0.7 | 150.30 | 85 |
| p08 | 11803.80 | 200 | 0.2 | 0.1 | 250 | 0.8 | 2665.07 | 147 |
| p09 | 12037.00 | 250 | 0.1 | 0.1 | 250 | 0.7 | 2928.28 | 141 |
| p10 | 12025.10 | 250 | 0.1 | 0.2 | 200 | 0.8 | 2149.62 | 9 |
| p11 | 11887.50 | 250 | 0.1 | 0.2 | 250 | 0.8 | 3744.39 | 234 |
| p15 | 7757.86 | 200 | 0.2 | 0.2 | 250 | 0.8 | 1062.48 | 8 |
| p18 | 15038.40 | 250 | 0.1 | 0.1 | 250 | 0.8 | 3408.23 | 101 |
| pr02 | 2188.68 | 250 | 0.2 | 0.2 | 150 | 0.7 | 350.92 | 1 |
| pr07 | 1671.37 | 250 | 0.1 | 0.1 | 150 | 0.7 | 405.52 | 71 |

## Mejor Configuracion Global

La mejor configuracion global (promediando rendimiento normalizado en todas las instancias):

```
population_size: 250.0
elite_percent: 0.1
mutants_percent: 0.2
max_generations: 250.0
p_bias: 0.8
Rendimiento normalizado: 0.1728
```

## Analisis de Sensibilidad

Impacto relativo de cada hiperparametro en el rendimiento (mayor porcentaje = mas influencia):

| Hiperparametro | Impacto (%) | Mejor Valor |
|---|---|---|
| population_size | 21.95% | 250 |
| elite_percent | 6.11% | 0.1 |
| mutants_percent | 5.88% | 0.1 |
| p_bias | 5.64% | 0.9 |
| max_generations | 1.37% | 250 |

### Recomendaciones de Valores

- **population_size**: Configurar en 250 (impacto alto: 21.9%)
- **elite_percent**: Preferiblemente usar 0.1 (impacto medio: 6.1%)
- **mutants_percent**: Preferiblemente usar 0.1 (impacto medio: 5.9%)
- **p_bias**: Preferiblemente usar 0.9 (impacto medio: 5.6%)
- max_generations: Usar 250 o cualquier valor cercano (impacto bajo: 1.4%)

## Analisis de Tiempo de Ejecucion

- Configuracion mas rapida: poblacion = 50.0, tiempo promedio = 98.27s
- Configuracion mas lenta: poblacion = 250.0, tiempo promedio = 416.39s
- Aceleracion potencial: 4.24x

## Analisis de Convergencia

- Convergencia mas rapida: poblacion = 50.0, generacion promedio = 26.4
- Convergencia mas lenta: poblacion = 250.0, generacion promedio = 33.7

## Conclusiones y Recomendaciones

1. El hiperparametro con mayor impacto en el rendimiento es **population_size**.
2. El hiperparametro con menor impacto es **max_generations**.

### Configuracion Recomendada

```
population_size: 250.0
elite_percent: 0.1
mutants_percent: 0.2
max_generations: 250.0
p_bias: 0.8
```

### Equilibrio entre Tiempo y Calidad

Para un equilibrio entre calidad de resultados y tiempo de ejecucion:

```
population_size: 250
elite_percent: 0.1
mutants_percent: 0.3
max_generations: 200
p_bias: 0.7
Tiempo estimado: 42.26s
Calidad relativa: 100.0%
```

### Recomendaciones por Tipo de Problema

Si necesitas optimizar para:

- **Problemas grandes**: Prioriza configuraciones con mayor tamanio de poblacion
- **Tiempo limitado**: Utiliza tamanios de poblacion mas pequenios con porcentajes de elites mas altos
- **Maxima calidad**: Utiliza la configuracion global recomendada

## Sugerencias para Futuras Optimizaciones

1. **Exploracion adaptativa**: Considerar implementar mecanismos adaptativos para los parametros mas sensibles
2. **Enfoque en population_size**: Realizar una busqueda mas fina para este parametro
3. **Hibridizacion**: Considerar combinar el algoritmo genetico con busquedas locales
4. **Paralelizacion**: Evaluar la implementacion de evaluacion paralela para poblaciones grandes
