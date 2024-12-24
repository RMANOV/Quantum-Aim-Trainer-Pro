import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
import time

from utils.logger import Logger
from config import QUANTUM_CONFIG


@dataclass
class OptimizationState:
    """Състояние на оптимизацията"""
    parameters: np.ndarray
    energy: float
    gradient: np.ndarray
    iteration: int
    convergence: float
    entanglement: float
    fidelity: float


class QuantumStateOptimizer:
    """Оптимизатор на квантови състояния за подобряване на игровата динамика"""

    def __init__(self):
        self.config = QUANTUM_CONFIG
        self.logger = Logger()

        # Невронна мрежа за оптимизация
        self.quantum_net = self._build_quantum_network()

        # Оптимизационни параметри
        self.learning_rate = self.config.learning_rate
        self.momentum = self.config.momentum
        self.iterations = self.config.max_iterations

        # История на оптимизацията
        self.optimization_history: List[OptimizationState] = []
        self.best_state: Optional[OptimizationState] = None

        # Квантови параметри
        self.hamiltonian = self._initialize_hamiltonian()
        self.current_state = self._initialize_quantum_state()

    def _build_quantum_network(self) -> nn.Module:
        """Създава квантово-вдъхновена невронна мрежа"""
        class QuantumNetwork(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.quantum_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                        nn.Dropout(0.2),

                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(),
                        nn.BatchNorm1d(hidden_size),

                        nn.Linear(hidden_size, input_size),
                        nn.Sigmoid()
                    ) for _ in range(3)  # Многослойна архитектура
                ])

                self.quantum_gate = nn.Parameter(
                    torch.randn(input_size, input_size, dtype=torch.complex64)
                )

            def forward(self, x):
                # Прилагане на класически слоеве
                for layer in self.quantum_layers:
                    x = layer(x)

                # Симулация на квантова трансформация
                x_complex = torch.complex(x, torch.zeros_like(x))
                x_transformed = torch.matmul(x_complex, self.quantum_gate)

                return torch.abs(x_transformed)

        return QuantumNetwork(
            self.config.state_size,
            self.config.hidden_size
        )

    def _initialize_hamiltonian(self) -> np.ndarray:
        """Инициализира Хамилтониан за квантова еволюция"""
        size = self.config.state_size
        H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        # Правим го Ермитов
        return (H + H.conj().T) / 2

    def _initialize_quantum_state(self) -> np.ndarray:
        """Инициализира начално квантово състояние"""
        state = np.random.randn(self.config.state_size) + \
            1j * np.random.randn(self.config.state_size)
        # Нормализация
        return state / np.linalg.norm(state)

    def optimize(self, target_parameters: Dict) -> OptimizationState:
        """Оптимизира квантовото състояние спрямо целевите параметри"""
        start_time = time.time()
        current_state = self.current_state

        for iteration in range(self.iterations):
            # Изчисляване на градиент
            gradient = self._calculate_gradient(
                current_state, target_parameters)

            # Обновяване на състоянието
            current_state = self._update_state(current_state, gradient)

            # Изчисляване на метрики
            energy = self._calculate_energy(current_state)
            convergence = self._calculate_convergence(gradient)
            entanglement = self._calculate_entanglement(current_state)
            fidelity = self._calculate_fidelity(
                current_state, target_parameters)

            # Запазване на състоянието
            state = OptimizationState(
                parameters=current_state,
                energy=energy,
                gradient=gradient,
                iteration=iteration,
                convergence=convergence,
                entanglement=entanglement,
                fidelity=fidelity
            )

            self.optimization_history.append(state)

            # Проверка за най-добро състояние
            if self.best_state is None or energy < self.best_state.energy:
                self.best_state = state

            # Проверка за сходимост
            if convergence < self.config.convergence_threshold:
                break

        self.logger.info(f"Optimization completed in {
                         time.time() - start_time:.2f}s")
        return self.best_state

    def _calculate_gradient(self, state: np.ndarray,
                            target_parameters: Dict) -> np.ndarray:
        """Изчислява градиент за оптимизация"""
        # Хамилтонов градиент
        energy_gradient = -1j * np.dot(self.hamiltonian, state)

        # Градиент спрямо целевите параметри
        target_gradient = self._target_gradient(state, target_parameters)

        return energy_gradient + target_gradient

    def _target_gradient(self, state: np.ndarray,
                         target_parameters: Dict) -> np.ndarray:
        """Изчислява градиент спрямо целевите параметри"""
        gradient = np.zeros_like(state)

        # Добавяне на принос от различни параметри
        if 'energy' in target_parameters:
            gradient += self._energy_gradient(state,
                                              target_parameters['energy'])

        if 'entanglement' in target_parameters:
            gradient += self._entanglement_gradient(state,
                                                    target_parameters['entanglement'])

        return gradient

    def _update_state(self, state: np.ndarray,
                      gradient: np.ndarray) -> np.ndarray:
        """Обновява квантовото състояние"""
        # Градиентна стъпка
        new_state = state - self.learning_rate * gradient

        # Добавяне на момент
        if len(self.optimization_history) > 0:
            previous_gradient = self.optimization_history[-1].gradient
            new_state -= self.momentum * previous_gradient

        # Нормализация
        return new_state / np.linalg.norm(new_state)

    def _calculate_energy(self, state: np.ndarray) -> float:
        """Изчислява енергията на състоянието"""
        return np.real(np.dot(state.conj(), np.dot(self.hamiltonian, state)))

    def _calculate_convergence(self, gradient: np.ndarray) -> float:
        """Изчислява мярка за сходимост"""
        return np.linalg.norm(gradient)

    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """Изчислява заплитането на състоянието"""
        # Опростен модел на заплитане чрез von Neumann entropy
        density_matrix = np.outer(state, state.conj())
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

    def _calculate_fidelity(self, state: np.ndarray,
                            target_parameters: Dict) -> float:
        """Изчислява близостта до целевото състояние"""
        if 'target_state' in target_parameters:
            target = target_parameters['target_state']
            overlap = np.abs(np.dot(state.conj(), target))**2
            return overlap
        return 1.0

    def get_optimization_metrics(self) -> Dict:
        """Връща метрики за оптимизацията"""
        if not self.optimization_history:
            return {}

        return {
            'final_energy': self.best_state.energy,
            'iterations': len(self.optimization_history),
            'convergence': self.best_state.convergence,
            'entanglement': self.best_state.entanglement,
            'fidelity': self.best_state.fidelity
        }
