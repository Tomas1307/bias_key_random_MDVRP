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
#include <set>
#include <sys/stat.h>  // Para verificar si un archivo existe
#include <sys/types.h>  // Para tipos de datos usados en stat
#include <direct.h>    // Para crear directorios en Windows

// Data structures for MDVRP
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

struct HyperparameterConfig {
    int population_size;
    double elite_percent;
    double mutant_percent;
    int max_generations;
    double p_bias;
    
    // For easy comparison in std::set
    bool operator<(const HyperparameterConfig& other) const {
        if (population_size != other.population_size) return population_size < other.population_size;
        if (elite_percent != other.elite_percent) return elite_percent < other.elite_percent;
        if (mutant_percent != other.mutant_percent) return mutant_percent < other.mutant_percent;
        if (max_generations != other.max_generations) return max_generations < other.max_generations;
        return p_bias < other.p_bias;
    }
    
    // String representation for logging
    std::string to_string() const {
        std::stringstream ss;
        ss << "pop=" << population_size 
           << "_elite=" << elite_percent 
           << "_mutant=" << mutant_percent 
           << "_gen=" << max_generations 
           << "_bias=" << p_bias;
        return ss.str();
    }
};

// Early stopping parameters
const int EARLY_STOPPING_BASE_FACTOR = 500;
const int EARLY_STOPPING_MIN = 10;
const int EARLY_STOPPING_MAX = 100;

// Random number generator
std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

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
        
        // Apply 2-opt to each route
        for (auto& route : improved_routes) {
            if (route.customers.size() <= 2) {
                continue;
            }
            
            auto depot_it = std::find_if(data.depots.begin(), data.depots.end(),
                                       [&route](const Depot& d) { return d.id == route.depot_id; });
            if (depot_it == data.depots.end()) {
                continue;
            }
            
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
                            break;
                        }
                    }
                    if (improved) break;
                }
            }
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
                int depot_id = it->first;
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
    std::pair<std::vector<Route>, double> solve(int generations = 100, bool verbose = false, int early_stopping = -1) {
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
        std::vector<Route> final_solution = final_result.first;
        double best_distance = final_result.second;
        
        // Return best solution and convergence generation
        return std::make_pair(final_solution, best_distance);
    }
    
    // Get convergence generation
    int get_convergence_generation(int generations = 100, bool verbose = false, int early_stopping = -1) {
        // Initialize
        auto population = initialize_population();
        double best_fitness = std::numeric_limits<double>::max();
        int no_improvement_count = 0;
        int convergence_gen = 0;
        
        // Calculate adaptive early stopping if not provided
        if (early_stopping < 0) {
            early_stopping = calculate_early_stopping(data.customers.size(), data.depots.size());
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
                convergence_gen = gen;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }
            
            // Early stopping
            if (no_improvement_count >= early_stopping) {
                break;
            }
            
            // Evolve population
            population = evolve(population, fitness_values);
        }
        
        return convergence_gen;
    }
    
    // Get depot usage statistics
    int count_used_depots(const std::vector<Route>& solution) {
        std::unordered_set<int> used_depots;
        for (const auto& route : solution) {
            used_depots.insert(route.depot_id);
        }
        return used_depots.size();
    }
};

int main(int argc, char* argv[]) {
    // Check if instance argument is provided
    if (argc < 2) {
        std::cerr << "Error: No instance name provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <instance_name>" << std::endl;
        std::cerr << "Example: " << argv[0] << " p01" << std::endl;
        return 1;
    }
    
    // Get instance name from command line
    std::string instance_name = argv[1];
    std::string instance_file = "./dat/" + instance_name + ".dat";
    
    // Check if file exists
    if (!file_exists(instance_file)) {
        std::cerr << "Error: Instance file not found: " << instance_file << std::endl;
        return 1;
    }
    
    // Define hyperparameter values to test
    std::vector<int> population_sizes = {250};
    std::vector<double> elite_percents = {0.1};
    std::vector<double> mutant_percents = {0.2};
    std::vector<int> max_generations_values = {250};
    std::vector<double> p_bias_values = {0.8};
    
    // Create set of all hyperparameter configurations
    std::set<HyperparameterConfig> all_configs;
    for (int pop_size : population_sizes) {
        for (double elite_pct : elite_percents) {
            for (double mutant_pct : mutant_percents) {
                for (int max_gen : max_generations_values) {
                    for (double p_bias : p_bias_values) {
                        all_configs.insert({pop_size, elite_pct, mutant_pct, max_gen, p_bias});
                    }
                }
            }
        }
    }
    
    // Create directory for results if it doesn't exist
    ensure_results_directory();
    
    // Create result file for this instance
    std::string result_filename = "results_cpp/" + instance_name + ".txt";
    std::ofstream result_file(result_filename, std::ios::app); // Append mode in case we're resuming
    
    if (!result_file.is_open()) {
        std::cerr << "Error: Could not open results file for writing: " << result_filename << std::endl;
        return 1;
    }
    
    // Write header if the file is new (empty)
    result_file.seekp(0, std::ios::end);
    if (result_file.tellp() == 0) {
        result_file << "# Hyperparameter Testing for Instance: " << instance_name << "\n";
        result_file << "# Format: config_string, best_distance, convergence_generation, execution_time, num_routes, num_used_depots\n";
        result_file << "#\n";
    }
    result_file.close(); // Close and reopen for each config to ensure data is saved
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "Processing instance: " << instance_name << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    // Parse instance
    MDVRPData data = parse_mdvrp_file(instance_file);
    if (data.customers.empty() || data.depots.empty()) {
        std::cerr << "Error: Failed to parse instance file or no customers/depots found." << std::endl;
        return 1;
    }
    
    std::cout << "Parsed instance with " << data.num_customers << " customers, " 
              << data.num_depots << " depots, " 
              << data.num_vehicles << " vehicles" << std::endl;
    
    // Test each hyperparameter configuration
    int config_index = 1;
    int total_configs = all_configs.size();
    
    for (const auto& config : all_configs) {
        std::string config_str = config.to_string();
        
        // Check if this config has already been tested by looking for it in the result file
        bool already_tested = false;
        std::ifstream check_file(result_filename);
        std::string line;
        while (std::getline(check_file, line)) {
            // If the line starts with the config string, it's already been tested
            if (line.find(config_str) == 0) {
                already_tested = true;
                break;
            }
        }
        check_file.close();
        
        if (already_tested) {
            std::cout << "Skipping configuration " << config_index << "/" << total_configs 
                      << " (already tested): " << config_str << std::endl;
            config_index++;
            continue;
        }
        
        std::cout << "Testing configuration " << config_index << "/" << total_configs 
                  << ": " << config_str << std::endl;
        
        // Create and run BRKGA with this configuration
        BRKGA_MDVRP brkga(data, config.population_size, config.elite_percent, 
                         config.mutant_percent, config.p_bias, true);
        
        // Measure execution time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Solve
        std::pair<std::vector<Route>, double> result = brkga.solve(config.max_generations, false);
        std::vector<Route> solution = result.first;
        double best_distance = result.second;
        
        // Get convergence generation
        int convergence_gen = brkga.get_convergence_generation(config.max_generations);
        
        // Calculate depot usage
        int num_used_depots = brkga.count_used_depots(solution);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count() / 1000.0;
        
        // Append result to the file
        std::ofstream append_file(result_filename, std::ios::app);
        if (append_file.is_open()) {
            append_file << config_str << ", " 
                       << best_distance << ", " 
                       << convergence_gen << ", " 
                       << execution_time << ", " 
                       << solution.size() << ", " 
                       << num_used_depots << "\n";
            append_file.close();
        } else {
            std::cerr << "Warning: Could not open results file for appending." << std::endl;
        }
        
        // Print result
        std::cout << "  Result: Distance=" << best_distance 
                  << ", Routes=" << solution.size() 
                  << ", Depots=" << num_used_depots
                  << ", Convergence=" << convergence_gen
                  << ", Time=" << execution_time << "s" << std::endl;
        
        config_index++;
    }
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Completed testing all configurations on instance: " << instance_name << std::endl;
    std::cout << "Results saved to: " << result_filename << std::endl;
    
    return 0;
}