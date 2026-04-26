"""
Тесты для VCG механизма
"""

import unittest
import numpy as np
from src.config import NodeConfig
from src.environment.edge_node import EdgeNode
from src.environment.task import Task
from src.mechanisms.auction import VCGAuctioneer

class TestVCGAuction(unittest.TestCase):
    """Тесты VCG аукциона"""
    
    def setUp(self):
        self.num_devices = 5
        self.num_nodes = 3
        self.auctioneer = VCGAuctioneer(self.num_devices, self.num_nodes)
    
    def test_auction_execution(self):
        """Тест выполнения аукциона"""
        # Создать случайные матрицы
        valuations = np.random.uniform(0.5, 1.0, (self.num_devices, self.num_nodes))
        costs = np.random.uniform(0.2, 0.5, (self.num_devices, self.num_nodes))
        
        # Провести аукцион
        result = self.auctioneer.run_auction(valuations, costs)
        
        # Проверить результаты
        self.assertIsNotNone(result.allocation)
        self.assertIsNotNone(result.payments)
        self.assertGreater(result.social_welfare, -1000)  # SW должна быть разумной
    
    def test_allocation_shape(self):
        """Тест формы матрицы распределения"""
        valuations = np.random.uniform(0.5, 1.0, (self.num_devices, self.num_nodes))
        costs = np.random.uniform(0.2, 0.5, (self.num_devices, self.num_nodes))
        
        result = self.auctioneer.run_auction(valuations, costs)
        
        self.assertEqual(result.allocation.shape, (self.num_devices, self.num_nodes))
        self.assertTrue(np.all(result.allocation.sum(axis=1) <= 1))
    
    def test_payments_positive(self):
        """Тест что платежи разумны"""
        valuations = np.ones((self.num_devices, self.num_nodes)) * 1.0
        costs = np.ones((self.num_devices, self.num_nodes)) * 0.3
        
        result = self.auctioneer.run_auction(valuations, costs)
        
        # Платежи должны быть в разумном диапазоне
        self.assertTrue(np.all(result.payments >= 0))
        self.assertTrue(np.all(result.payments <= 10))

    def test_exact_allocation_with_resource_constraints(self):
        """Точный поиск должен находить глобально выгодное распределение."""
        auctioneer = VCGAuctioneer(num_devices=2, num_nodes=2)
        utilities = np.array([[10.0, 9.0], [9.0, 1.0]])
        costs = np.zeros_like(utilities)
        tasks = [
            Task(id=0, device_id=0, cpu_required=5, memory_required=1, data_size=1),
            Task(id=1, device_id=1, cpu_required=5, memory_required=1, data_size=1),
        ]
        nodes = [
            EdgeNode(0, NodeConfig(cpu_capacity=5, memory_capacity=8)),
            EdgeNode(1, NodeConfig(cpu_capacity=5, memory_capacity=8)),
        ]

        result = auctioneer.run_auction(utilities, costs, tasks=tasks, nodes=nodes)

        expected_allocation = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(result.allocation, expected_allocation)
        self.assertAlmostEqual(result.social_welfare, 18.0)

    def test_vcg_payment_uses_net_contribution(self):
        """VCG-платёж должен учитывать внешний эффект по value minus cost."""
        auctioneer = VCGAuctioneer(num_devices=2, num_nodes=1)
        utilities = np.array([[10.0], [8.0]])
        costs = np.array([[6.0], [1.0]])
        tasks = [
            Task(id=0, device_id=0, cpu_required=1, memory_required=1, data_size=1),
            Task(id=1, device_id=1, cpu_required=1, memory_required=1, data_size=1),
        ]
        nodes = [EdgeNode(0, NodeConfig(cpu_capacity=1, memory_capacity=4))]

        result = auctioneer.run_auction(utilities, costs, tasks=tasks, nodes=nodes)

        np.testing.assert_array_equal(result.allocation, np.array([[0], [1]]))
        self.assertAlmostEqual(result.payments[1], 4.0)
        self.assertAlmostEqual(result.payments[0], 0.0)

if __name__ == '__main__':
    unittest.main()
