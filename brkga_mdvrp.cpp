#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <ctime>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

// Algorithm parameters - Fixed configuration
const int POPULATION_SIZE = 250;    
const double ELITE_PERCENT = 0.1;
const double MUTANTS_PERCENT = 0.2;
const int MAX_GENERATIONS = 250;
const double P_BIAS = 0.8;
const bool USE_REFINEMENT = true;

// Early stopping parameters
const int EARLY_STOPPING_BASE_FACTOR = 500;
const int EARLY_STOPPING_MIN = 10;
const int EARLY_STOPPING_MAX = 100;

// Random number generator
std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

// Data structures
struct Customer {
    int id;
    double x;
    double y;
    double service_duration;
    double demand;
    int frequency;
    int num_visit_combinations;
    std::vector<int> visit_combinations;
    std::pair<double, double> time_window; // (early, late)
};

struct Depot {
    int id;
    double x;
    double y;
};

struct VehicleInfo {
    double max_duration;
    double max_load;
};

struct MDVRPData {
    int problem_type;
    int num_vehicles;
    int num_customers;
    int num_depots;
    std::vector<Depot> depots;
    std::vector<Customer> customers;
    std::vector<VehicleInfo> vehicle_info;
};

struct Route {
    int depot_id;
    std::vector<int> customers;
    double load;
    double distance;
};

// Función para verificar si un archivo existe
bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Función para crear un directorio
bool create_directory(const std::string& directory) {
    #ifdef _WIN32
    return _mkdir(directory.c_str()) == 0;
    #else
    return mkdir(directory.c_str(), 0777) == 0;
    #endif
}

// Función para asegurar que exista el directorio de resultados
void ensure_results_directory() {
    if (!file_exists("results_cpp")) {
        if (create_directory("results_cpp")) {
            std::cout << "Created directory: results_cpp" << std::endl;
        } else {
            std::cerr << "Error: Could not create results directory." << std::endl;
        }
    }
}

// Función para verificar si una instancia ya ha sido procesada
bool instance_already_processed(const std::string& instance_name) {
    std::string result_filename = "results_cpp/results.txt";
    if (!file_exists(result_filename)) {
        return false;
    }
    
    std::ifstream result_file(result_filename);
    if (!result_file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(result_file, line)) {
        // Si la línea comienza con el nombre de la instancia, ya ha sido procesada
        if (line.find("Instance: " + instance_name) != std::string::npos) {
            result_file.close();
            return true;
        }
    }
    
    result_file.close();
    return false;
}

// Function to calculate Euclidean distance
double euclidean_distance(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

// Function to calculate route distance
double calculate_route_distance(const Depot& depot, const std::vector<int>& customers, const std::vector<Customer>& all_customers) {
    if (customers.empty()) {
        return 0.0;
    }

    double distance = 0.0;
    double prev_x = depot.x;
    double prev_y = depot.y;

    for (int cust_id : customers) {
        // Find customer by ID
        auto it = std::find_if(all_customers.begin(), all_customers.end(),
                            [cust_id](const Customer& c) { return c.id == cust_id; });
        if (it == all_customers.end()) {
            continue;  // Skip if customer not found
        }

        // Add distance from previous point
        distance += euclidean_distance(prev_x, prev_y, it->x, it->y);
        
        // Update previous coordinates
        prev_x = it->x;
        prev_y = it->y;
    }

    // Return to depot
    distance += euclidean_distance(prev_x, prev_y, depot.x, depot.y);
    
    return distance;
}

// Calculate adaptive early stopping value
int calculate_early_stopping(int num_customers, int num_depots) {
    int problem_size = num_customers + num_depots;
    int early_stop = EARLY_STOPPING_BASE_FACTOR / problem_size;
    
    return std::max(EARLY_STOPPING_MIN, std::min(early_stop, EARLY_STOPPING_MAX));
}

// Parse MDVRP file
MDVRPData parse_mdvrp_file(const std::string& file_path) {
    MDVRPData data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return data;
    }
    
    std::string line;
    
    // Read first line (problem type, m, n, t)
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> data.problem_type >> data.num_vehicles >> data.num_customers >> data.num_depots;
    }
    
    // Read vehicle information
    for (int i = 0; i < data.num_depots; i++) {
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            double D, Q;
            iss >> D >> Q;
            data.vehicle_info.push_back({D, Q});
        }
    }
    
    // Read customer information
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int id;
        double x, y, service_duration, demand;
        int frequency, num_visit_combinations;
        
        iss >> id >> x >> y >> service_duration >> demand >> frequency >> num_visit_combinations;
        
        std::vector<int> visit_combinations;
        for (int i = 0; i < num_visit_combinations; i++) {
            int comb;
            iss >> comb;
            visit_combinations.push_back(comb);
        }
        
        // Check if this is a depot (demand = 0, frequency = 0)
        if (demand == 0 && frequency == 0) {
            data.depots.push_back({id, x, y});
        } else {
            // Read time window if available
            double early = 0, late = 0;
            if (iss >> early >> late) {
                data.customers.push_back({id, x, y, service_duration, demand, frequency, 
                                        num_visit_combinations, visit_combinations, {early, late}});
            } else {
                data.customers.push_back({id, x, y, service_duration, demand, frequency, 
                                        num_visit_combinations, visit_combinations, {0, 0}});
            }
        }
    }
    
    // If no depots were found, the first several customers might be depots
    if (data.depots.empty() && !data.customers.empty()) {
        // The first num_depots entries might be depots
        int potential_depots = std::min(data.num_depots, static_cast<int>(data.customers.size()));
        
        for (int i = 0; i < potential_depots; i++) {
            data.depots.push_back({data.customers[i].id, data.customers[i].x, data.customers[i].y});
        }
        
        // Remove these from customers
        if (potential_depots > 0) {
            data.customers.erase(data.customers.begin(), data.customers.begin() + potential_depots);
        }
    }
    
    return data;
}

class BRKGA_MDVRP {
private:
    MDVRPData data;
    int population_size;
    int elite_size;
    int mutants_size;
    int num_genes;
    double p_bias; // Probability of inheriting from elite parent
    bool use_refinement;
    
    // Random number generators
    std::uniform_real_distribution<double> dist_0_1;
    
public:
    BRKGA_MDVRP(const MDVRPData& data, int pop_size = 100, double elite_pct = 0.2, 
                double mutants_pct = 0.1, double bias = 0.7, bool use_refine = true) 
        : data(data), 
          population_size(pop_size), 
          elite_size(static_cast<int>(pop_size * elite_pct)),
          mutants_size(static_cast<int>(pop_size * mutants_pct)),
          p_bias(bias),
          use_refinement(use_refine),
          dist_0_1(0.0, 1.0)
    {
        // Number of genes: 2 genes per customer (assignment + order)
        num_genes = 2 * data.customers.size();
    }
    
    // Initialize population randomly
    std::vector<std::vector<double>> initialize_population() {
        std::vector<std::vector<double>> population(population_size, std::vector<double>(num_genes));
        
        for (int i = 0; i < population_size; i++) {
            for (int j = 0; j < num_genes; j++) {
                population[i][j] = dist_0_1(rng);
            }
        }
        
        return population;
    }
    
    // Crossover operator
    std::vector<double> crossover(const std::vector<double>& elite_parent, 
                                 const std::vector<double>& non_elite_parent) {
        std::vector<double> child(num_genes);
        
        for (int i = 0; i < num_genes; i++) {
            child[i] = (dist_0_1(rng) < p_bias) ? elite_parent[i] : non_elite_parent[i];
        }
        
        return child;
    }
    
    // Evolve population
    std::vector<std::vector<double>> evolve(const std::vector<std::vector<double>>& population, 
                                           const std::vector<double>& fitness_values) {
        // Get sorted indices
        std::vector<int> indices(population_size);
        for (int i = 0; i < population_size; i++) {
            indices[i] = i;
        }
        
        std::sort(indices.begin(), indices.end(), 
                 [&fitness_values](int i, int j) { return fitness_values[i] < fitness_values[j]; });
        
        // Extract elite
        std::vector<std::vector<double>> elite;
        for (int i = 0; i < elite_size; i++) {
            elite.push_back(population[indices[i]]);
        }
        
        // Extract non-elite
        std::vector<std::vector<double>> non_elite;
        for (int i = elite_size; i < population_size; i++) {
            non_elite.push_back(population[indices[i]]);
        }
        
        // Create new population
        std::vector<std::vector<double>> new_population;
        
        // Add elite
        for (const auto& ind : elite) {
            new_population.push_back(ind);
        }
        
        // Add offspring
        std::uniform_int_distribution<int> elite_dist(0, elite_size - 1);
        std::uniform_int_distribution<int> non_elite_dist(0, non_elite.size() - 1);
        
        for (int i = 0; i < population_size - elite_size - mutants_size; i++) {
            int elite_idx = elite_dist(rng);
            int non_elite_idx = non_elite_dist(rng);
            
            new_population.push_back(crossover(elite[elite_idx], non_elite[non_elite_idx]));
        }
        
        // Add mutants
        for (int i = 0; i < mutants_size; i++) {
            std::vector<double> mutant(num_genes);
            for (int j = 0; j < num_genes; j++) {
                mutant[j] = dist_0_1(rng);
            }
            new_population.push_back(mutant);
        }
        
        return new_population;
    }
    
    // Decode chromosome to solution
    std::pair<std::vector<Route>, double> decode(const std::vector<double>& chromosome) {
        // Add occasional progress indicator for decode operations
        static int decode_count = 0;
        if (++decode_count % 1000 == 0) {
            std::cout << "Processed " << decode_count << " chromosome decodings" << std::endl;
        }
        
        int num_customers = data.customers.size();
        int num_depots = data.depots.size();
        
        // 1. Assign customers to depots
        std::vector<int> depot_assignments(num_customers);
        for (int i = 0; i < num_customers; i++) {
            int depot_idx = static_cast<int>(chromosome[i] * num_depots);
            depot_idx = std::min(depot_idx, num_depots - 1);
            depot_assignments[i] = depot_idx;
        }
        
        // 2. Sort customers within each depot
        std::vector<int> sorted_indices(num_customers);
        for (int i = 0; i < num_customers; i++) {
            sorted_indices[i] = i;
        }
        
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                 [&chromosome, num_customers](int i, int j) { 
                     return chromosome[num_customers + i] < chromosome[num_customers + j]; 
                 });
        
        // 3. Build initial routes
        std::unordered_map<int, std::vector<int>> routes;
        for (int depot_id = 0; depot_id < num_depots; depot_id++) {
            routes[data.depots[depot_id].id] = std::vector<int>();
        }
        
        for (int idx : sorted_indices) {
            int depot_id = data.depots[depot_assignments[idx]].id;
            int customer_id = data.customers[idx].id;
            routes[depot_id].push_back(customer_id);
        }
        
        // 4. Split into feasible routes
        std::vector<Route> feasible_routes;
        double total_distance = 0.0;
        
        // Iterate through each depot and its customers
        for (auto it = routes.begin(); it != routes.end(); ++it) {
            int depot_id = it->first;
            const std::vector<int>& customers = it->second;
            
            // Find depot
            auto depot_it = std::find_if(data.depots.begin(), data.depots.end(),
                                       [depot_id](const Depot& d) { return d.id == depot_id; });
            if (depot_it == data.depots.end()) {
                continue;
            }
            
            std::vector<int> current_route;
            double current_load = 0.0;
            double current_duration = 0.0;
            
            for (int cust_id : customers) {
                // Find customer
                auto cust_it = std::find_if(data.customers.begin(), data.customers.end(),
                                          [cust_id](const Customer& c) { return c.id == cust_id; });
                if (cust_it == data.customers.end()) {
                    continue;
                }
                
                // Check if adding this customer would violate constraints
                if (current_load + cust_it->demand > data.vehicle_info[0].max_load ||
                    current_duration + cust_it->service_duration > data.vehicle_info[0].max_duration) {
                    
                    // Create new route from current_route
                    if (!current_route.empty()) {
                        double route_distance = calculate_route_distance(*depot_it, current_route, data.customers);
                        feasible_routes.push_back({depot_id, current_route, current_load, route_distance});
                        total_distance += route_distance;
                        
                        current_route.clear();
                        current_load = 0.0;
                        current_duration = 0.0;
                    }
                }
                
                // Add customer to current route
                current_route.push_back(cust_id);
                current_load += cust_it->demand;
                current_duration += cust_it->service_duration;
            }
            
            // Add final route if not empty
            if (!current_route.empty()) {
                double route_distance = calculate_route_distance(*depot_it, current_route, data.customers);
                feasible_routes.push_back({depot_id, current_route, current_load, route_distance});
                total_distance += route_distance;
            }
        }
        
        // 5. Apply refinement if enabled
        if (use_refinement) {
            std::pair<std::vector<Route>, double> refined = refine_solution(feasible_routes);
            return refined;
        }
        
        return std::make_pair(feasible_routes, total_distance);
    }
    
    // Refine solution
    std::pair<std::vector<Route>, double> refine_solution(const std::vector<Route>& routes) {
        std::vector<Route> improved_routes = routes;
        int route_count = 0;
        int total_improvements = 0;
        
        // Apply 2-opt to each route
        for (auto& route : improved_routes) {
            route_count++;
            
            if (route.customers.size() <= 2) {
                continue;
            }
            
            auto depot_it = std::find_if(data.depots.begin(), data.depots.end(),
                                       [&route](const Depot& d) { return d.id == route.depot_id; });
            if (depot_it == data.depots.end()) {
                continue;
            }
            
            int improvements = 0;
            bool improved = true;
            int iteration = 0;
            
            // Limit maximum iterations to prevent excessive processing
            const int MAX_ITERATIONS = 10;
            
            while (improved && iteration < MAX_ITERATIONS) {
                iteration++;
                improved = false;
                double best_distance = route.distance;
                
                for (size_t i = 0; i < route.customers.size() - 1; i++) {
                    for (size_t j = i + 1; j < route.customers.size(); j++) {
                        // Create a new route with segments reversed
                        std::vector<int> new_customers = route.customers;
                        std::reverse(new_customers.begin() + i, new_customers.begin() + j + 1);
                        
                        double new_distance = calculate_route_distance(*depot_it, new_customers, data.customers);
                        
                        if (new_distance < best_distance) {
                            best_distance = new_distance;
                            route.customers = new_customers;
                            route.distance = new_distance;
                            improved = true;
                            improvements++;
                            break;
                        }
                    }
                    if (improved) break;
                }
            }
            
            total_improvements += improvements;
        }
        
        // Calculate total distance
        double total_distance = 0.0;
        for (const auto& route : improved_routes) {
            total_distance += route.distance;
        }
        
        return std::make_pair(improved_routes, total_distance);
    }
    
    // Fitness function
    double fitness(const std::vector<double>& chromosome) {
        // Add a progress indicator for fitness evaluations
        static int eval_count = 0;
        if (++eval_count % 1000 == 0) {
            std::cout << "Processed " << eval_count << " fitness evaluations" << std::endl;
        }
        
        std::pair<std::vector<Route>, double> result = decode(chromosome);
        std::vector<Route> solution = result.first;
        double total_distance = result.second;
        
        // Factors to consider in evaluation
        int num_routes = solution.size();
        std::unordered_set<int> used_depots;
        for (const auto& route : solution) {
            used_depots.insert(route.depot_id);
        }
        int num_used_depots = used_depots.size();
        
        // Penalty for not using all depots
        double depot_penalty = (num_used_depots < data.num_depots) ? 
                              1000.0 * (data.num_depots - num_used_depots) : 0.0;
        
        // Penalty for excessive vehicles
        double vehicle_penalty = 100.0 * std::max(0, num_routes - data.num_vehicles);
        
        // Penalty for depot load imbalance
        double balance_penalty = 0.0;
        if (num_used_depots > 0) {
            std::unordered_map<int, int> routes_per_depot;
            for (const auto& route : solution) {
                routes_per_depot[route.depot_id]++;
            }
            
            // Calculate standard deviation
            double mean_routes = static_cast<double>(num_routes) / num_used_depots;
            double variance = 0.0;
            
            for (auto it = routes_per_depot.begin(); it != routes_per_depot.end(); ++it) {
                int count = it->second;
                variance += std::pow(count - mean_routes, 2);
            }
            variance /= num_used_depots;
            double std_dev = std::sqrt(variance);
            
            balance_penalty = 50.0 * std_dev;
        }
        
        // Final fitness (minimization)
        return total_distance + depot_penalty + vehicle_penalty + balance_penalty;
    }
    
    // Solve MDVRP
    std::pair<std::vector<Route>, double> solve(int generations = 100, bool verbose = true, int early_stopping = -1) {
        // Initialize
        auto population = initialize_population();
        double best_fitness = std::numeric_limits<double>::max();
        std::vector<Route> best_solution;
        std::vector<double> best_chromosome;
        int no_improvement_count = 0;
        int convergence_gen = 0;
        
        // Calculate adaptive early stopping if not provided
        if (early_stopping < 0) {
            early_stopping = calculate_early_stopping(data.customers.size(), data.depots.size());
        }
        
        if (verbose) {
            std::cout << "Starting BRKGA with population=" << population_size 
                      << ", elite%=" << (elite_size / (double)population_size)
                      << ", mutant%=" << (mutants_size / (double)population_size)
                      << ", p_bias=" << p_bias
                      << ", refinement=" << (use_refinement ? "enabled" : "disabled") << std::endl;
            std::cout << "Early stopping: " << early_stopping << " generations without improvement" << std::endl;
        }
        
        // Main loop
        for (int gen = 0; gen < generations; gen++) {
            // Evaluate population
            std::vector<double> fitness_values(population_size);
            for (int i = 0; i < population_size; i++) {
                fitness_values[i] = fitness(population[i]);
            }
            
            // Find best solution
            int best_idx = std::min_element(fitness_values.begin(), fitness_values.end()) - fitness_values.begin();
            double current_fitness = fitness_values[best_idx];
            
            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_chromosome = population[best_idx];
                std::pair<std::vector<Route>, double> decoded_result = decode(best_chromosome);
                best_solution = decoded_result.first;
                convergence_gen = gen;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }
            
            // Print progress if verbose
            if (verbose && gen % 10 == 0) {
                std::cout << "Gen " << gen << "/" << generations << ": Fitness = " << best_fitness;
                
                if (no_improvement_count > 0) {
                    std::cout << ", No improvement: " << no_improvement_count << "/" << early_stopping;
                }
                
                std::cout << std::endl;
            }
            
            // Early stopping
            if (no_improvement_count >= early_stopping) {
                if (verbose) {
                    std::cout << "Early stopping at generation " << gen 
                              << " - " << no_improvement_count << " generations without improvement" << std::endl;
                }
                break;
            }
            
            // Evolve population
            population = evolve(population, fitness_values);
        }
        
        // Get final solution
        std::pair<std::vector<Route>, double> final_result = decode(best_chromosome);
        
        return final_result;
    }
    
    // Get information about depots usage
    int count_used_depots(const std::vector<Route>& solution) {
        std::unordered_set<int> used_depots;
        for (const auto& route : solution) {
            used_depots.insert(route.depot_id);
        }
        return used_depots.size();
    }
};

// Function to get instance name from file path
std::string get_instance_name(const std::string& file_path) {
    // Extract file name without extension
    std::string file_name;
    size_t last_slash = file_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        file_name = file_path.substr(last_slash + 1);
    } else {
        file_name = file_path;
    }
    
    size_t last_dot = file_name.find_last_of(".");
    if (last_dot != std::string::npos) {
        return file_name.substr(0, last_dot);
    }
    
    return file_name;
}

// Function to list all files in a directory
std::vector<std::string> list_files_in_directory(const std::string& directory) {
    std::vector<std::string> files;
    
    #ifdef _WIN32
    WIN32_FIND_DATA file_data;
    HANDLE dir = FindFirstFile((directory + "\\*").c_str(), &file_data);
    if (dir != INVALID_HANDLE_VALUE) {
        do {
            if (!(file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(directory + "\\" + file_data.cFileName);
            }
        } while (FindNextFile(dir, &file_data));
        FindClose(dir);
    }
    #else
    // Usar dirent.h para sistemas POSIX (Linux/Unix/MacOS)
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            // Solo añadir archivos regulares (no directorios)
            std::string full_path = directory + "/" + filename;
            struct stat stat_buf;
            if (stat(full_path.c_str(), &stat_buf) == 0 && S_ISREG(stat_buf.st_mode)) {
                files.push_back(full_path);
            }
        }
        closedir(dir);
    }
    #endif
    
    return files;
}

int main(int argc, char* argv[]) {
    // Asegurar que exista el directorio de resultados
    ensure_results_directory();
    
    // Crear o abrir el archivo de resultados
    std::string result_filename = "results_cpp/results.txt";
    bool new_file = !file_exists(result_filename);
    
    std::ofstream result_file(result_filename, std::ios::app); // Modo añadir
    if (!result_file.is_open()) {
        std::cerr << "Error: Could not open results file for writing: " << result_filename << std::endl;
        return 1;
    }
    
    // Escribir encabezado si el archivo es nuevo
    if (new_file) {
        result_file << "# BRKGA MDVRP Results\n";
        result_file << "# Configuration: population=" << POPULATION_SIZE 
                    << ", elite%=" << ELITE_PERCENT
                    << ", mutant%=" << MUTANTS_PERCENT
                    << ", generations=" << MAX_GENERATIONS
                    << ", p_bias=" << P_BIAS
                    << ", refinement=" << (USE_REFINEMENT ? "enabled" : "disabled") << "\n";
        result_file << "# Format: Instance, Customers, Depots, Vehicles, Best_Fitness, Routes, Used_Depots, Early_Stop, Convergence_Gen, Execution_Time(s)\n";
        result_file << "#\n";
    }
    result_file.close();
    
    // Listar todos los archivos de instancia en la carpeta dat
    std::vector<std::string> instance_files = list_files_in_directory("dat");
    std::cout << "Found " << instance_files.size() << " instance files in the dat directory." << std::endl;
    
    // Procesar cada instancia
    int count = 0;
    for (const std::string& file_path : instance_files) {
        // Obtener nombre de la instancia
        std::string instance_name = get_instance_name(file_path);
        
        // Verificar si la instancia ya ha sido procesada
        if (instance_already_processed(instance_name)) {
            std::cout << "Skipping instance " << instance_name << " (already processed)" << std::endl;
            continue;
        }
        
        std::cout << "\n=======================================================" << std::endl;
        std::cout << "Processing instance " << (++count) << "/" << instance_files.size() 
                  << ": " << instance_name << std::endl;
        std::cout << "=======================================================" << std::endl;
        
        // Parse instance file
        MDVRPData data = parse_mdvrp_file(file_path);
        
        if (data.customers.empty() || data.depots.empty()) {
            std::cerr << "Error: Failed to parse instance file or no customers/depots found." << std::endl;
            continue;
        }
        
        std::cout << "Instance info: " << data.num_customers << " customers, " 
                  << data.num_depots << " depots, " 
                  << data.num_vehicles << " vehicles" << std::endl;
        
        // Set up and run BRKGA
        std::cout << "Setting up BRKGA with fixed parameters and running algorithm..." << std::endl;
        
        BRKGA_MDVRP brkga(data, POPULATION_SIZE, ELITE_PERCENT, MUTANTS_PERCENT, P_BIAS, USE_REFINEMENT);
        
        // Measure execution time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run the algorithm
        std::pair<std::vector<Route>, double> result = brkga.solve(MAX_GENERATIONS, true);
        std::vector<Route> solution = result.first;
        double best_fitness = result.second;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count() / 1000.0;
        
        // Calculate metrics
        int num_routes = solution.size();
        int num_used_depots = brkga.count_used_depots(solution);
        int early_stopping = calculate_early_stopping(data.num_customers, data.num_depots);
        
        // Get the convergence generation (not available directly from solve method, needs additional tracking)
        // For simplicity, we'll use -1 as a placeholder
        int convergence_gen = -1;
        
        // Display results
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Best fitness: " << best_fitness << std::endl;
        std::cout << "  Number of routes: " << num_routes << std::endl;
        std::cout << "  Depots used: " << num_used_depots << "/" << data.num_depots << std::endl;
        std::cout << "  Execution time: " << execution_time << " seconds" << std::endl;
        
        // Save results to file
        result_file.open(result_filename, std::ios::app);
        if (result_file.is_open()) {
            result_file << "Instance: " << instance_name << ", "
                        << data.num_customers << ", "
                        << data.num_depots << ", "
                        << data.num_vehicles << ", "
                        << best_fitness << ", "
                        << num_routes << ", "
                        << num_used_depots << ", "
                        << early_stopping << ", "
                        << convergence_gen << ", "
                        << execution_time << "\n";
            result_file.close();
        } else {
            std::cerr << "Warning: Could not open results file for appending." << std::endl;
        }
    }
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "All instances processed. Results saved to: " << result_filename << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    return 0;
}