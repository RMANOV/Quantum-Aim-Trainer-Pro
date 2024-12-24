import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import math
import time

from utils.logger import Logger
from config import TARGET_CONFIG


@dataclass
class Target:
    """Структура за мишена"""
    position: np.ndarray
    velocity: np.ndarray
    size: float
    color: Tuple[int, int, int, int]
    health: int
    is_fake: bool
    pattern: str
    properties: Dict
    creation_time: float
    hits_required: int
    current_hits: int


class TargetSystem:
    """Система за управление на мишените с комплексно поведение"""

    def __init__(self):
        self.config = TARGET_CONFIG
        self.logger = Logger()

        # Колекции от мишени
        self.targets: List[Target] = []
        self.target_patterns: Dict[str, callable] = self._initialize_patterns()

        # Системни параметри
        self.difficulty = 1.0
        self.spawn_timer = 0.0
        self.last_spawn = time.time()

        # Квантови параметри
        self.quantum_phase = 0.0
        self.quantum_state = np.random.random(4)

        # Метрики
        self.targets_spawned = 0
        self.targets_destroyed = 0
        self.accuracy_history = []

    def _initialize_patterns(self) -> Dict[str, callable]:
        """Инициализира шаблоните за движение"""
        return {
            'linear': self._linear_pattern,
            'sine': self._sine_pattern,
            'circular': self._circular_pattern,
            'pursuit': self._pursuit_pattern,
            'quantum': self._quantum_pattern,
            'erratic': self._erratic_pattern,
            'spiral': self._spiral_pattern,
            'bounce': self._bounce_pattern
        }

    def update(self, delta_time: float, player_position: Tuple[float, float]):
        """Обновява системата от мишени"""
        self.spawn_timer += delta_time

        # Обновяване на квантовото състояние
        self._update_quantum_state(delta_time)

        # Проверка за създаване на нови мишени
        self._check_spawn()

        # Обновяване на съществуващите мишени
        self._update_targets(delta_time, player_position)

        # Проверка за колизии между мишени
        self._check_collisions()

    def _update_quantum_state(self, delta_time: float):
        """Обновява квантовото състояние на системата"""
        self.quantum_phase += delta_time * 2.0

        # Симулация на квантова еволюция
        rotation = np.array([
            [np.cos(self.quantum_phase), -np.sin(self.quantum_phase)],
            [np.sin(self.quantum_phase), np.cos(self.quantum_phase)]
        ])

        self.quantum_state = np.dot(rotation, self.quantum_state[:2])
        self.quantum_state = np.concatenate([
            self.quantum_state,
            np.random.random(2) * 0.1  # квантов шум
        ])

    def spawn_target(self, position: Optional[Tuple[float, float]] = None):
        """Създава нова мишена"""
        if position is None:
            position = self._get_spawn_position()

        # Определяне на параметрите базирано на трудността
        size = self.config.base_size * (1.0 - self.difficulty * 0.3)
        health = max(1, int(self.difficulty * 2))
        is_fake = random.random() < self.difficulty * 0.2

        # Избор на шаблон за движение
        pattern = random.choice(list(self.target_patterns.keys()))
        if random.random() < self.difficulty * 0.3:
            pattern = 'quantum'  # По-голям шанс за квантово поведение при висока трудност

        target = Target(
            position=np.array(position),
            velocity=np.zeros(2),
            size=size,
            color=self._generate_target_color(is_fake),
            health=health,
            is_fake=is_fake,
            pattern=pattern,
            properties={
                'phase': random.uniform(0, 2 * np.pi),
                'frequency': random.uniform(1, 3) * self.difficulty,
                'amplitude': random.uniform(30, 100),
                'quantum_state': self.quantum_state.copy()
            },
            creation_time=time.time(),
            hits_required=max(1, int(self.difficulty * 1.5)),
            current_hits=0
        )

        self.targets.append(target)
        self.targets_spawned += 1

    def _get_spawn_position(self) -> Tuple[float, float]:
        """Определя позиция за създаване на мишена"""
        if random.random() < 0.3:
            # Квантово-базирано позициониране
            x = self.config.screen_width * (0.5 + self.quantum_state[0])
            y = self.config.screen_height * (0.5 + self.quantum_state[1])
        else:
            # Стандартно позициониране по края на екрана
            if random.random() < 0.5:
                x = random.choice([0, self.config.screen_width])
                y = random.uniform(0, self.config.screen_height)
            else:
                x = random.uniform(0, self.config.screen_width)
                y = random.choice([0, self.config.screen_height])

        return (x, y)

    def _update_targets(self, delta_time: float, player_position: Tuple[float, float]):
        """Обновява всички мишени"""
        for target in self.targets[:]:
            # Прилагане на шаблон за движение
            pattern_func = self.target_patterns.get(target.pattern)
            if pattern_func:
                pattern_func(target, delta_time, player_position)

            # Обновяване на позицията
            target.position += target.velocity * delta_time

            # Проверка за излизане от екрана
            if self._is_out_of_bounds(target.position):
                self.targets.remove(target)

    def _quantum_pattern(self, target: Target, delta_time: float,
                         player_position: Tuple[float, float]):
        """Квантово-подобно движение"""
        # Изчисляване на квантова вълнова функция
        phase = target.properties['phase'] + \
            delta_time * target.properties['frequency']
        quantum_pos = np.array([
            math.sin(phase + target.position[1] * 0.01),
            math.cos(phase + target.position[0] * 0.01)
        ])

        # Смесване с класическо движение
        classical_vel = self._get_classical_velocity(target, player_position)
        quantum_influence = 0.3 + 0.7 * self.difficulty

        target.velocity = (classical_vel * (1 - quantum_influence) +
                           quantum_pos * target.properties['amplitude'] * quantum_influence)
        target.properties['phase'] = phase

    def _get_classical_velocity(self, target: Target,
                                player_position: Tuple[float, float]) -> np.ndarray:
        """Изчислява класическата компонента на скоростта"""
        to_player = np.array(player_position) - target.position
        distance = np.linalg.norm(to_player)

        if distance > 0:
            return to_player / distance * 100.0
        return np.zeros(2)

    def process_hit(self, target_index: int) -> bool:
        """Обработва попадение в мишена"""
        if 0 <= target_index < len(self.targets):
            target = self.targets[target_index]

            if target.is_fake:
                return False

            target.current_hits += 1

            if target.current_hits >= target.hits_required:
                self.targets.remove(target)
                self.targets_destroyed += 1
                self.accuracy_history.append(True)
                return True

        return False

    def get_metrics(self) -> Dict:
        """Връща метрики за системата"""
        return {
            'active_targets': len(self.targets),
            'spawned': self.targets_spawned,
            'destroyed': self.targets_destroyed,
            'accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0.0,
            'quantum_complexity': np.linalg.norm(self.quantum_state)
        }
