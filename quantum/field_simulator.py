import numpy as np
from scipy import fft, signal
import pygame
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

from utils.logger import Logger
from config import QUANTUM_CONFIG


@dataclass
class QuantumState:
    """Квантово състояние с вълнова функция"""
    amplitude: np.ndarray
    phase: np.ndarray
    coherence: float
    entanglement: float
    energy: float


class QuantumFieldSimulator:
    """Симулатор на квантово поле за непредсказуемо поведение на мишените"""

    def __init__(self, screen_size: Tuple[int, int]):
        self.config = QUANTUM_CONFIG
        self.logger = Logger()

        self.screen_size = screen_size
        self.grid_size = (screen_size[0] // self.config.cell_size,
                          screen_size[1] // self.config.cell_size)

        # Инициализация на квантовото поле
        self.field = self._initialize_field()
        self.previous_field = np.copy(self.field.amplitude)

        # Параметри на симулацията
        self.time = 0.0
        self.dt = self.config.time_step
        self.coherence_history = []

        # Визуализация
        self.visualization_surface = pygame.Surface(screen_size)
        self.colormap = plt.cm.viridis

        # Квантови характеристики
        self.interference_points = []
        self.vortices = []
        self.quantum_noise = self._generate_quantum_noise()

    def _initialize_field(self) -> QuantumState:
        """Инициализира квантовото поле"""
        amplitude = np.random.normal(0, 1, self.grid_size) + \
            1j * np.random.normal(0, 1, self.grid_size)
        phase = np.angle(amplitude)

        # Нормализация
        amplitude /= np.sqrt(np.sum(np.abs(amplitude)**2))

        return QuantumState(
            amplitude=amplitude,
            phase=phase,
            coherence=1.0,
            entanglement=0.0,
            energy=self._calculate_energy(amplitude)
        )

    def evolve(self, delta_time: float):
        """Развива квантовото поле във времето"""
        self.time += delta_time
        steps = int(delta_time / self.dt)

        for _ in range(steps):
            self._evolution_step()

        # Обновяване на характеристиките
        self._update_quantum_characteristics()

    def _evolution_step(self):
        """Един времеви стъп на еволюцията"""
        # Запазване на предишното състояние
        self.previous_field = np.copy(self.field.amplitude)

        # Прилагане на квантов оператор на еволюция
        k_space = fft.fft2(self.field.amplitude)
        k_space *= np.exp(-1j * self.config.potential_strength * self.dt)
        self.field.amplitude = fft.ifft2(k_space)

        # Добавяне на квантов шум
        self.field.amplitude += self.quantum_noise * np.random.normal(0,
                                                                      self.config.noise_strength, self.grid_size)

        # Нормализация
        self.field.amplitude /= np.sqrt(
            np.sum(np.abs(self.field.amplitude)**2))

        # Обновяване на фазата
        self.field.phase = np.angle(self.field.amplitude)

    def _update_quantum_characteristics(self):
        """Обновява квантовите характеристики на полето"""
        # Изчисляване на кохерентност
        self.field.coherence = self._calculate_coherence()

        # Изчисляване на заплитане
        self.field.entanglement = self._calculate_entanglement()

        # Изчисляване на енергия
        self.field.energy = self._calculate_energy(self.field.amplitude)

        # Откриване на вихри
        self._detect_vortices()

        # Откриване на интерференчни точки
        self._detect_interference_points()

    def _calculate_coherence(self) -> float:
        """Изчислява кохерентността на полето"""
        correlation = np.abs(np.sum(np.conj(self.previous_field) *
                                    self.field.amplitude))
        norm = np.sqrt(np.sum(np.abs(self.previous_field)**2) *
                       np.sum(np.abs(self.field.amplitude)**2))
        return correlation / norm if norm > 0 else 0

    def _calculate_entanglement(self) -> float:
        """Изчислява квантовото заплитане"""
        # Опростен модел на заплитане
        rho = np.outer(self.field.amplitude.flatten(),
                       np.conj(self.field.amplitude.flatten()))
        eigenvalues = np.linalg.eigvalsh(rho)
        return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

    def _calculate_energy(self, wavefunction: np.ndarray) -> float:
        """Изчислява енергията на полето"""
        kinetic = -0.5 * np.sum(np.conj(wavefunction) *
                                self._laplacian(wavefunction))
        potential = np.sum(np.abs(wavefunction)**2 * self._potential())
        return np.real(kinetic + potential)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Изчислява Лапласиан на полето"""
        return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
                np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field) / \
               (self.config.cell_size**2)

    def _potential(self) -> np.ndarray:
        """Генерира потенциално поле"""
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        X, Y = np.meshgrid(x, y)
        return self.config.potential_strength * (X**2 + Y**2)

    def _detect_vortices(self):
        """Открива квантови вихри в полето"""
        self.vortices = []
        phase_gradient = np.gradient(self.field.phase)

        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                circulation = (phase_gradient[0][i, j]**2 +
                               phase_gradient[1][i, j]**2)
                if circulation > self.config.vortex_threshold:
                    self.vortices.append((i, j))

    def _detect_interference_points(self):
        """Открива точки на интерференция"""
        self.interference_points = []
        amplitude = np.abs(self.field.amplitude)**2

        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                if (amplitude[i, j] > self.config.interference_threshold and
                        amplitude[i, j] > amplitude[i-1:i+2, j-1:j+2].mean()):
                    self.interference_points.append((i, j))

    def render(self) -> pygame.Surface:
        """Визуализира квантовото поле"""
        # Получаване на интензитета
        intensity = np.abs(self.field.amplitude)**2

        # Нормализация за визуализация
        normalized = (intensity - intensity.min()) / \
            (intensity.max() - intensity.min() + 1e-10)

        # Преобразуване в цветове
        colors = self.colormap(normalized)
        colors = (colors[:, :, :3] * 255).astype(np.uint8)

        # Създаване на pygame повърхност
        surface = pygame.surfarray.make_surface(colors.transpose(1, 0, 2))

        # Добавяне на вихри и интерференчни точки
        for vortex in self.vortices:
            pos = (int(vortex[0] * self.config.cell_size),
                   int(vortex[1] * self.config.cell_size))
            pygame.draw.circle(surface, (255, 0, 0), pos, 3)

        for point in self.interference_points:
            pos = (int(point[0] * self.config.cell_size),
                   int(point[1] * self.config.cell_size))
            pygame.draw.circle(surface, (0, 255, 0), pos, 3)

        return surface

    def get_field_value(self, position: Tuple[float, float]) -> complex:
        """Връща стойността на полето в дадена точка"""
        x = int(position[0] / self.config.cell_size) % self.grid_size[0]
        y = int(position[1] / self.config.cell_size) % self.grid_size[1]
        return self.field.amplitude[x, y]

    def get_coherence(self) -> float:
        """Връща текущата кохерентност на полето"""
        return self.field.coherence
