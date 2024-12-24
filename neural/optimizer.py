import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam
import time

from utils.logger import Logger
from config import OPTIMIZER_CONFIG


@dataclass
class OptimizationMetrics:
    loss_history: List[float]
    convergence_rate: float
    optimization_time: float
    improvement_rate: float
    stability_score: float


class QuantumInspiredOptimizer:
    """Хибриден оптимизатор, комбиниращ квантови принципи с невронни мрежи"""

    def __init__(self):
        self.config = OPTIMIZER_CONFIG
        self.logger = Logger()

        # Параметри на оптимизацията
        self.population_size = self.config.population_size
        self.quantum_amplitude = self.config.quantum_amplitude
        self.learning_rate = self.config.learning_rate

        # Невронна мрежа за оптимизация
        self.network = self._build_network()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate)

        # Метрики
        self.metrics = OptimizationMetrics(
            loss_history=[],
            convergence_rate=0.0,
            optimization_time=0.0,
            improvement_rate=0.0,
            stability_score=0.0
        )

        # Състояние на оптимизацията
        self.current_generation = 0
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.quantum_states = self._initialize_quantum_states()

    def _build_network(self) -> nn.Module:
        """Създава невронна мрежа за оптимизация"""
        class OptimizationNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.2),

                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.2),

                    nn.Linear(hidden_size, output_size),
                    nn.Tanh()
                )

            def forward(self, x):
                return self.network(x)

        return OptimizationNetwork(
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size
        )

    def _initialize_quantum_states(self) -> np.ndarray:
        """Инициализира квантово-подобни състояния"""
        states = np.zeros((self.population_size, self.config.state_size),
                          dtype=np.complex128)

        for i in range(self.population_size):
            # Създаване на суперпозиция от състояния
            amplitude = np.random.uniform(0, self.quantum_amplitude)
            phase = np.random.uniform(0, 2 * np.pi)
            states[i] = amplitude * np.exp(1j * phase)

        return states

    def optimize(self, fitness_function, max_iterations: int = 100) -> Dict:
        """Основен метод за оптимизация"""
        start_time = time.time()

        for iteration in range(max_iterations):
            # Квантова фаза
            quantum_solutions = self._quantum_phase()

            # Невронна фаза
            neural_solutions = self._neural_phase(quantum_solutions)

            # Оценка на решенията
            fitness_values = [fitness_function(
                sol) for sol in neural_solutions]

            # Обновяване на най-доброто решение
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = neural_solutions[best_idx]

            # Обновяване на квантовите състояния
            self._update_quantum_states(fitness_values)

            # Обучение на невронната мрежа
            self._train_network(quantum_solutions,
                                neural_solutions, fitness_values)

            # Обновяване на метрики
            self._update_metrics(fitness_values, iteration)

        self.metrics.optimization_time = time.time() - start_time
        return self._get_optimization_results()

    def _quantum_phase(self) -> np.ndarray:
        """Квантова фаза на оптимизация"""
        solutions = np.zeros((self.population_size, self.config.state_size))

        for i in range(self.population_size):
            # Квантово измерване
            state = self.quantum_states[i]
            probability = np.abs(state) ** 2

            # Колапс на квантовото състояние
            solutions[i] = np.random.normal(0, probability)

        return solutions

    def _neural_phase(self, quantum_solutions: np.ndarray) -> np.ndarray:
        """Невронна фаза на оптимизация"""
        with torch.no_grad():
            solutions = torch.tensor(quantum_solutions, dtype=torch.float32)
            refined_solutions = self.network(solutions)
            return refined_solutions.numpy()

    def _train_network(self, quantum_solutions: np.ndarray,
                       neural_solutions: np.ndarray,
                       fitness_values: List[float]):
        """Обучение на невронната мрежа"""
        # Подготовка на данните
        X = torch.tensor(quantum_solutions, dtype=torch.float32)
        y = torch.tensor(neural_solutions, dtype=torch.float32)
        weights = torch.tensor(fitness_values, dtype=torch.float32)
        weights = weights / weights.sum()  # Нормализация

        # Обучение
        self.optimizer.zero_grad()
        predictions = self.network(X)
        loss = nn.MSELoss()(predictions, y) * weights.unsqueeze(1)
        loss.backward()
        self.optimizer.step()

        self.metrics.loss_history.append(loss.item())

    def _update_quantum_states(self, fitness_values: List[float]):
        """Обновява квантовите състояния базирано на фитнес стойностите"""
        fitness_array = np.array(fitness_values)
        normalized_fitness = (fitness_array - fitness_array.min()) / \
            (fitness_array.max() - fitness_array.min() + 1e-10)

        for i in range(self.population_size):
            # Ротация на фазата базирана на фитнес стойността
            phase_shift = 2 * np.pi * normalized_fitness[i]
            self.quantum_states[i] *= np.exp(1j * phase_shift)

            # Нормализация
            self.quantum_states[i] /= np.abs(self.quantum_states[i])

    def _update_metrics(self, fitness_values: List[float], iteration: int):
        """Обновява метриките на оптимизацията"""
        if iteration > 0:
            current_mean = np.mean(fitness_values)
            previous_mean = np.mean(
                self.metrics.loss_history[-self.population_size:])
            self.metrics.improvement_rate = (
                current_mean - previous_mean) / previous_mean

        self.metrics.convergence_rate = self._calculate_convergence_rate()
        self.metrics.stability_score = self._calculate_stability_score()

    def _get_optimization_results(self) -> Dict:
        """Връща резултатите от оптимизацията"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'metrics': self.metrics,
            'generations': self.current_generation,
            'quantum_states': self.quantum_states
        }
