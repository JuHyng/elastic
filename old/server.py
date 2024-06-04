import flwr as fl
from collections import defaultdict

class FlowerServer(fl.server.Server):
    def __init__(self, total_bandwidth, noise_power_density, client_manager):
        super().__init__(client_manager=client_manager)
        self.total_bandwidth = total_bandwidth
        self.noise_power_density = noise_power_density
        self.global_model = None  # Initialize your global model
        self.client_resources = {}  # Store resources like CPU frequency and power for each client

    def fit(self, ins, config):
        # Implement your training logic here
        # Decide on client selection based on P1, P2, P3 optimization steps
        selected_clients = self.select_clients(ins)
        results = self.train_clients(selected_clients)
        self.global_model = self.aggregate_results(results)
        return self.global_model

    def evaluate(self, ins, config):
        # Request each client to evaluate the global model
        evaluation_results = [client.evaluate(self.global_model.get_weights(), config) for client in ins]

        # Aggregate results
        aggregated_results = self.aggregate_evaluation_results(evaluation_results)
        return aggregated_results

    def aggregate_evaluation_results(self, results):
        # Example aggregation by averaging accuracy
        total_accuracy = sum(result['accuracy'] for result in results)
        average_accuracy = total_accuracy / len(results)
        return {"average_accuracy": average_accuracy}

    def select_clients(self, ins):
        # Example data structure for client info
        clients_info = {client.cid: {'f_i': client.config['f_max'], 'p_i': client.config['p_max']} for client in ins}

        # Compute ni for each client
        for cid, info in clients_info.items():
            info['n_i'] = self.compute_ni(info['f_i'], info['p_i'])  # Assume compute_ni is implemented

        # Sort clients based on ni
        sorted_clients = sorted(clients_info.items(), key=lambda x: x[1]['n_i'])

        # Container for selected clients
        selected_clients = []
        current_latency = 0

        # Iterate over sorted clients and select based on latency
        for cid, info in sorted_clients:
            comp_latency, wait_time, up_time = self.estimate_latencies(info['f_i'], info['p_i'])  # Assume this function is available
            total_latency = comp_latency + wait_time + up_time

            # Check if adding this client exceeds the time threshold
            if current_latency + total_latency <= self.tau:  # tau is the latency deadline
                selected_clients.append(cid)
                current_latency += total_latency
            else:
                break

        return selected_clients

    def compute_ni(self, f_i, p_i):
        # Dummy implementation, replace with actual logic
        return f_i * p_i  # Simplified for demonstration

    def estimate_latencies(self, f_i, p_i):
        # Dummy implementation, replace with actual calculations based on the model
        comp_latency = f_i / 1000  # Simplified for demonstration
        wait_time = f_i / 2000  # Simplified for demonstration
        up_time = p_i / 1000  # Simplified for demonstration
        return comp_latency, wait_time, up_time


    def aggregate_results(self, results):
        total_data_count = sum(result["num_examples"] for result in results)
        new_global_weights = None
        
        # Iterate over each result and aggregate weighted updates
        for result in results:
            client_weights, num_examples = result['model_weights'], result['num_examples']
            client_weight_factor = num_examples / total_data_count
            
            if new_global_weights is None:
                new_global_weights = [weights * client_weight_factor for weights in client_weights]
            else:
                for i, weights in enumerate(client_weights):
                    new_global_weights[i] += weights * client_weight_factor
        
        # Setting the updated weights as the new global model weights
        self.global_model.set_weights(new_global_weights)
        return self.global_model.get_weights()


    def start_server(self):
        # Start the Flower server with custom configurations
        fl.server.start_server()
        
    # return config (address, port, etc.)
    def get_config(self):
        return self.config
    
