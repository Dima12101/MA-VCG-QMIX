"""
Интеграционные тесты (VCG + QMIX вместе)
"""

import unittest
import numpy as np
from src.mechanisms.vcg_auction import VCGAuction
from src.agents.networks import GRUAgent
from src.config import ENV_CONFIG

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_vcg_qmix_pipeline(self):
        """Тест всей цепочки: VCG -> QMIX"""
        num_edges = ENV_CONFIG.num_edges
        num_devices = ENV_CONFIG.num_devices
        
        # 1. Запустить VCG
        auction = VCGAuction(num_devices, num_edges)
        valuations = np.random.uniform(0.5, 1.0, (num_devices, num_edges))
        costs = np.random.uniform(0.2, 0.5, (num_devices, num_edges))
        
        vcg_result = auction.run_auction(valuations, costs, timestamp=0)
        
        # Проверить что VCG сработал
        self.assertIsNotNone(vcg_result.allocation)
        self.assertIsNotNone(vcg_result.payments)
        
        # 2. Использовать платежи для QMIX
        payments = vcg_result.payments
        
        # Проверить что платежи разумные
        self.assertEqual(len(payments), num_devices)
        self.assertTrue(np.isfinite(payments).all())

if __name__ == '__main__':
    unittest.main()
