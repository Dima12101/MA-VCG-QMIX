import numpy as np
from typing import Tuple
from .auction import VCGAuctioneer

def calculate_vcg_payments(
    allocation: np.ndarray,  # Binary matrix [m x n]: allocation[i][j] = 1 if device i uses edge j
    utility_matrix: np.ndarray,  # [m x n]: U[i][j]
    cost_matrix: np.ndarray,  # [m x n]: C[i][j]
) -> Tuple[np.ndarray, float]:
    """
    Вычислить платежи VCG
    
    Args:
        allocation: матрица размещения [m x n]
        utility_matrix: матрица полезности [m x n]
        cost_matrix: матрица стоимости [m x n]
    
    Returns:
        payments: вектор платежей [m]
        total_sw: итоговое социальное благосустояние
    """
    num_devices, num_nodes = allocation.shape
    auctioneer = VCGAuctioneer(num_devices, num_nodes)
    social_welfare = auctioneer.compute_social_welfare(utility_matrix, cost_matrix, allocation)
    payments = auctioneer._compute_vcg_payments(
        allocation,
        utility_matrix,
        cost_matrix,
        current_sw=social_welfare,
    )
    return payments, social_welfare
