from server import FlowerServer
import flwr as fl
from client import FlowerClient


# server.py
if __name__ == "__main__":
    client_manager = fl.server.SimpleClientManager()
        
    server = FlowerServer(total_bandwidth= 5e6, noise_power_density=1e-9, client_manager=client_manager)
    server.start_server()
    
    print(server.config)
    
    # create 30 clients
    for i in range(30):
        client = FlowerClient(cid=i, model=None, device=None, data=None, config=None)
        
        # connect the client to the server
        client.connect("grpc://localhost:8080")
        print(f"Client {i} connected to the server.")
    
