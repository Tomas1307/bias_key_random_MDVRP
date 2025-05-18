import json
import re
import csv
from typing import Dict, List, Set, Any

def procesar_resultados_json(json_file_path: str, csv_output_path: str = None) -> str:
    """
    Procesa un archivo JSON con resultados de optimización y genera un CSV con información resumida.
    
    Args:
        json_file_path: Ruta al archivo JSON de resultados
        csv_output_path: Ruta donde guardar el CSV (opcional)
        
    Returns:
        Contenido del CSV generado
    """
    try:
        # Leer el archivo
        with open(json_file_path, 'r') as file:
            file_content = file.read()
        
        resultados = []
        
        # Intentar parsear como JSON completo
        try:
            data = json.loads(file_content)
            
            # Procesar cada instancia en el JSON
            for instance_key, instance in data.items():
                resultados.append(procesar_instancia(instance))
                
        # Si falla el parseo, extraer manualmente
        except json.JSONDecodeError:
            print("Error al parsear JSON completo, extrayendo manualmente...")
            
            # Extraer información directamente con expresiones regulares
            instance_match = re.search(r'"instance":\s*"([^"]+)"', file_content)
            fitness_match = re.search(r'"fitness":\s*([\d.]+)', file_content)
            execution_time_match = re.search(r'"execution_time":\s*([\d.]+)', file_content)
            
            if instance_match and fitness_match and execution_time_match:
                instancia = instance_match.group(1)
                fitness = float(fitness_match.group(1))
                tiempo = float(execution_time_match.group(1))
                
                # Contar depósitos únicos
                depot_ids = set()
                for match in re.finditer(r'"depot_id":\s*(\d+)', file_content):
                    depot_ids.add(match.group(1))
                depots = len(depot_ids)
                
                # Contar rutas (vehicles/cars)
                route_matches = re.findall(r'"depot_id":', file_content)
                cars = len(route_matches) if route_matches else 0
                
                # Encontrar la carga máxima
                max_capacity = 0
                for match in re.finditer(r'"load":\s*([\d.]+)', file_content):
                    load = float(match.group(1))
                    if load > max_capacity:
                        max_capacity = load
                
                resultados.append({
                    'instancia': instancia,
                    'tiempo': tiempo,
                    'fitness': fitness,
                    'depots': depots,
                    'cars': cars,
                    'max_capacity': max_capacity
                })
            else:
                raise ValueError("No se pudieron extraer los datos requeridos del archivo")
        
        # Generar el CSV
        headers = ["tiempo", "fitness", "instancia", "depots", "cars", "max_capacity"]
        
        # Crear contenido CSV en memoria
        csv_content = ','.join(headers) + '\n'
        for result in resultados:
            row = [
                str(result['tiempo']),
                str(result['fitness']),
                result['instancia'],
                str(result['depots']),
                str(result['cars']),
                str(result['max_capacity'])
            ]
            csv_content += ','.join(row) + '\n'
        
        # Opcionalmente guardar a archivo
        if csv_output_path:
            with open(csv_output_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                for result in resultados:
                    writer.writerow([
                        result['tiempo'],
                        result['fitness'],
                        result['instancia'],
                        result['depots'],
                        result['cars'],
                        result['max_capacity']
                    ])
            
        return csv_content
        
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        raise


def procesar_instancia(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa una instancia individual del JSON
    
    Args:
        instance: Objeto de instancia del JSON
        
    Returns:
        Objeto con la información procesada
    """
    # Obtener número de depósitos (depots)
    depot_ids = {route['depot_id'] for route in instance['best_solution']}
    depots = len(depot_ids)
    
    # Obtener número de vehículos (cars)
    cars = len(instance['best_solution'])
    
    # Calcular capacidad máxima (estimación basada en las cargas)
    loads = [route['load'] for route in instance['best_solution']]
    max_capacity = max(loads)
    
    # Crear y devolver registro para esta instancia
    return {
        'instancia': instance['instance'],
        'tiempo': instance['execution_time'],
        'fitness': instance['fitness'],
        'depots': depots,
        'cars': cars,
        'max_capacity': max_capacity
    }


# Ejemplo de uso
if __name__ == "__main__":
    csv_content = procesar_resultados_json('./resultados.json', './resumen_resultados.csv')
    print(f"CSV generado:\n{csv_content}")
    pass