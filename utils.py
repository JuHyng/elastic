# utils.py
import torch
import numpy as np

class LatencyModel(torch.nn.Module):
    def __init__(self, B, N0, s):
        super().__init__()
        # B: Total bandwidth available at the base station
        # N0: Background noise and intercell interference power density
        # s: Size of each local model in bits
        self.B = B
        self.N0 = N0
        self.s = s

    def forward(self, C_i, D_i, f_i, P_i, g_i, order_index, current_indices):
        # C_i: Number of CPU cycles required per sample
        # D_i: Number of training data samples
        # f_i: CPU frequency of client i
        # P_i: Transmission power of client i
        # g_i: Channel gain from client i to the BS
        # order_index: Order index of the client based on computational latency
        # current_indices: List of indices of clients currently uploading
        
        # Computational Latency
        theta = 0.1  # a constant to be tuned or defined according to system specifications
        epsilon = 0.01  # desired accuracy, also needs definition based on requirements
        comp_latency = theta * np.log2(1/epsilon) * C_i * D_i / f_i

        # Upload Latency
        r_i = self.B * np.log2(1 + P_i * g_i / self.N0)
        upload_latency = self.s / r_i

        # Waiting Time
        if order_index in current_indices:
            current_max_latency = max([comp_latency[j] + upload_latency[j] for j in current_indices if j < order_index])
            wait_latency = max(0, current_max_latency - comp_latency[order_index] - upload_latency[order_index])
        else:
            wait_latency = 0

        total_latency = comp_latency + upload_latency + wait_latency
        return total_latency, comp_latency, upload_latency, wait_latency

class EnergyModel(torch.nn.Module):
    def __init__(self, B, N0):
        super().__init__()
        # B: Total bandwidth available at the base station
        # N0: Background noise and intercell interference power density
        self.B = B
        self.N0 = N0

    def forward(self, C_i, D_i, f_i, P_i, g_i, s):
        # C_i: Number of CPU cycles required per sample
        # D_i: Number of training data samples
        # f_i: CPU frequency of client i
        # P_i: Transmission power of client i
        # g_i: Channel gain from client i to the BS
        # s: Size of each local model in bits
        
        # Constants
        theta = 0.1  # Example value, needs to be defined according to the system requirements
        epsilon = 0.01  # Desired accuracy, also needs definition based on requirements

        # Energy Consumption for Computing Local Model
        E_comp = (theta * np.log2(1 / epsilon) * C_i * D_i * f_i**2)

        # Upload Latency
        r_i = self.B * np.log2(1 + P_i * g_i / self.N0)
        upload_latency = s / r_i

        # Energy Consumption of Uploading Local Model
        E_upload = P_i * upload_latency

        return E_comp, E_upload, E_comp + E_upload
