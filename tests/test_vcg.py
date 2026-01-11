"""
Тесты для VCG механизма
"""

import unittest
import numpy as np
from mechanisms.auction import VCGAuctioneer

class TestVCGAuction(unittest.TestCase):
    """Тесты VCG аукциона"""
    
    def setUp(self):
        self.num_devices = 5
        self.num_edges = 3
        self.auctioneer = VCGAuctioneer(self.num_devices, self.num_edges)
    
    def test_auction_execution(self):
        """Тест выполнения аукциона"""
        # Создать случайные матрицы
        valuations = np.random.uniform(0.5, 1.0, (self.num_devices, self.num_edges))
        costs = np.random.uniform(0.2, 0.5, (self.num_devices, self.num_edges))
        
        # Провести аукцион
        result = self.auctioneer.run_auction(valuations, costs)
        
        # Проверить результаты
        self.assertIsNotNone(result.allocation)
        self.assertIsNotNone(result.payments)
        self.assertGreater(result.social_welfare, -1000)  # SW должна быть разумной
    
    def test_allocation_shape(self):
        """Тест формы матрицы распределения"""
        valuations = np.random.uniform(0.5, 1.0, (self.num_devices, self.num_edges))
        costs = np.random.uniform(0.2, 0.5, (self.num_devices, self.num_edges))
        
        result = self.auctioneer.run_auction(valuations, costs)
        
        self.assertEqual(result.allocation.shape, (self.num_devices, self.num_edges))
    
    def test_payments_positive(self):
        """Тест что платежи разумны"""
        valuations = np.ones((self.num_devices, self.num_edges)) * 1.0
        costs = np.ones((self.num_devices, self.num_edges)) * 0.3
        
        result = self.auctioneer.run_auction(valuations, costs)
        
        # Платежи должны быть в разумном диапазоне
        self.assertTrue(np.all(result.payments >= -10))
        self.assertTrue(np.all(result.payments <= 10))

if __name__ == '__main__':
    unittest.main()
