import flwr as fl
from utils import LatencyModel, EnergyModel

class FlowerClient(fl.client.Client):
    def __init__(self, cid, model, device, data, config):
        super().__init__(cid)
        self.model = model
        self.device = device
        self.data = data
        self.config = config  # includes C_i, f_i, P_i, g_i, etc.

    def fit(self, parameters, config):
        # Update local model with received global parameters
        self.model.set_weights(parameters)
        
        # Perform local training
        self.train_local_model()
        
        # Compute energy and latency
        energy_model = EnergyModel(self.config['B'], self.config['N0'])
        latency_model = LatencyModel(self.config['B'], self.config['N0'], self.config['s'])
        energy_consumed, _ = energy_model.forward(self.config['C_i'], len(self.data), self.config['f_i'], self.config['P_i'], self.config['g_i'], self.config['s'])
        latency, _, _, _ = latency_model.forward(self.config['C_i'], len(self.data), self.config['f_i'], self.config['P_i'], self.config['g_i'], self.client_index, self.current_indices)
        
        # Return the updated parameters and additional metrics
        return self.model.get_weights(), {"energy": energy_consumed, "latency": latency}

    def train_local_model(self):
        # Implement your training logic here
        pass

    def evaluate(self, parameters, config):
        # Load local validation data
        local_validation_data = self.load_validation_data()  # This needs to be implemented

        # Set global model weights
        self.model.set_weights(parameters)

        # Compute evaluation metrics, e.g., accuracy
        accuracy = self.model.test_on_data(local_validation_data)
        return {'accuracy': accuracy}