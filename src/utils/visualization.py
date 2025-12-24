"""
Утилиты для визуализации
"""

import matplotlib.pyplot as plt
import numpy as np

class NetworkVisualizer:
    """Визуализация edge-сети"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_network_topology(
        self,
        num_edges: int,
        num_devices: int,
        node_loads: np.ndarray = None
    ):
        """
        Нарисовать топологию сети
        
        Args:
            num_edges: количество edge-узлов
            num_devices: количество устройств
            node_loads: нагрузка на каждый узел (опционально)
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Нарисовать edge-узлы (в горизонтальной линии внизу)
        edge_x = np.linspace(0, 10, num_edges)
        edge_y = np.zeros(num_edges)
        
        colors = plt.cm.RdYlGn_r(node_loads / node_loads.max()) if node_loads is not None else 'blue'
        
        self.ax.scatter(edge_x, edge_y, s=1000, c=colors, marker='s', label='Edge nodes', zorder=3)
        
        # Нарисовать устройства (случайные позиции)
        device_x = np.random.uniform(0, 10, num_devices)
        device_y = np.random.uniform(5, 15, num_devices)
        self.ax.scatter(device_x, device_y, s=100, c='orange', marker='o', label='Devices', zorder=2)
        
        # Связи (от каждого устройства к ближайшему узлу)
        for i in range(num_devices):
            closest_edge = np.argmin(np.abs(edge_x - device_x[i]))
            self.ax.plot([device_x[i], edge_x[closest_edge]], [device_y[i], edge_y[closest_edge]], 
                        'k-', alpha=0.1, zorder=1)
        
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-2, 16)
        self.ax.set_xlabel('X position', fontsize=11)
        self.ax.set_ylabel('Y position', fontsize=11)
        self.ax.set_title('Edge Network Topology', fontsize=13)
        self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        return self.fig
