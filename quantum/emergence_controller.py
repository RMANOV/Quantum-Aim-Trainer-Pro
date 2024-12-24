import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import deque
import time

from utils.logger import Logger
from config import QUANTUM_CONFIG


@dataclass
class EmergentPattern:
    """Структура за съхранение на емерджентни модели"""
    pattern_id: int
    strength: float
    stability: float
    influence_radius: float
    lifetime: float
    properties: Dict
    affected_entities: List[int]


class EmergenceController:
    """Контролер за емерджентно поведение в квантовата система"""

    def __init__(self):
        self.config = QUANTUM_CONFIG
        self.logger = Logger()

        # Емерджентни модели
        self.active_patterns: List[EmergentPattern] = []
        self.pattern_history = deque(maxlen=1000)

        # Невронна мрежа за разпознаване на модели
        self.pattern_recognizer = self._build_pattern_recognizer()

        # Състояние на системата
        self.complexity = 0.0
        self.coherence = 1.0
        self.entropy = 0.0

        # Времеви параметри
        self.time = 0.0
        self.last_update = time.time()

    def _build_pattern_recognizer(self) -> nn.Module:
        """Създава невронна мрежа за разпознаване на емерджентни модели"""
        class PatternRecognizer(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_patterns: int):
                super().__init__()

                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.3),

                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.Dropout(0.2)
                )

                self.pattern_head = nn.Sequential(
                    nn.Linear(hidden_size // 2, num_patterns),
                    nn.Softmax(dim=1)
                )

                self.property_head = nn.Sequential(
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size // 4, 3)  # стабилност, сила, радиус
                )

            def forward(self, x):
                features = self.encoder(x)
                patterns = self.pattern_head(features)
                properties = self.property_head(features)
                return patterns, properties

        return PatternRecognizer(
            self.config.state_size,
            self.config.hidden_size,
            self.config.num_patterns
        )

    def update(self, quantum_state: np.ndarray, delta_time: float):
        """Обновява емерджентното поведение"""
        self.time += delta_time

        # Обновяване на системните параметри
        self._update_system_parameters(quantum_state)

        # Откриване на нови модели
        self._detect_patterns(quantum_state)

        # Обновяване на активните модели
        self._update_patterns(delta_time)

        # Взаимодействие между моделите
        self._process_pattern_interactions()

        self.last_update = time.time()

    def _update_system_parameters(self, quantum_state: np.ndarray):
        """Обновява параметрите на системата"""
        # Изчисляване на комплексност
        self.complexity = self._calculate_complexity(quantum_state)

        # Изчисляване на кохерентност
        self.coherence = self._calculate_coherence(quantum_state)

        # Изчисляване на ентропия
        self.entropy = self._calculate_entropy(quantum_state)

    def _detect_patterns(self, quantum_state: np.ndarray):
        """Открива нови емерджентни модели"""
        # Подготовка на данните
        state_tensor = torch.FloatTensor(quantum_state).unsqueeze(0)

        # Разпознаване на модели
        with torch.no_grad():
            patterns, properties = self.pattern_recognizer(state_tensor)

            # Анализ на резултатите
            pattern_probs = patterns[0].numpy()
            pattern_props = properties[0].numpy()

            # Създаване на нови модели
            for i, prob in enumerate(pattern_probs):
                if prob > self.config.pattern_threshold:
                    self._create_pattern(i, prob, pattern_props)

    def _create_pattern(self, pattern_id: int, strength: float,
                        properties: np.ndarray):
        """Създава нов емерджентен модел"""
        pattern = EmergentPattern(
            pattern_id=pattern_id,
            strength=strength,
            stability=properties[0],
            influence_radius=properties[1],
            lifetime=properties[2] * self.config.max_pattern_lifetime,
            properties={
                'creation_time': self.time,
                'complexity_contribution': self.complexity * strength,
                'coherence_factor': self.coherence * strength
            },
            affected_entities=[]
        )

        self.active_patterns.append(pattern)
        self.pattern_history.append(pattern)

    def _update_patterns(self, delta_time: float):
        """Обновява активните емерджентни модели"""
        for pattern in self.active_patterns[:]:
            # Обновяване на времето на живот
            pattern.lifetime -= delta_time

            # Проверка за изтекъл живот
            if pattern.lifetime <= 0:
                self.active_patterns.remove(pattern)
                continue

            # Обновяване на силата
            decay_rate = 1.0 - (delta_time / self.config.pattern_decay_time)
            pattern.strength *= decay_rate

            # Премахване на слаби модели
            if pattern.strength < self.config.min_pattern_strength:
                self.active_patterns.remove(pattern)

    def _process_pattern_interactions(self):
        """Обработва взаимодействията между моделите"""
        if len(self.active_patterns) < 2:
            return

        for i, pattern1 in enumerate(self.active_patterns):
            for pattern2 in self.active_patterns[i+1:]:
                self._interact_patterns(pattern1, pattern2)

    def _interact_patterns(self, pattern1: EmergentPattern,
                           pattern2: EmergentPattern):
        """Обработва взаимодействие между два модела"""
        # Изчисляване на взаимодействието
        interaction_strength = min(pattern1.strength, pattern2.strength)

        # Проверка за припокриване
        if self._patterns_overlap(pattern1, pattern2):
            # Конструктивна или деструктивна интерференция
            if self._is_constructive(pattern1, pattern2):
                self._constructive_interaction(pattern1, pattern2,
                                               interaction_strength)
            else:
                self._destructive_interaction(pattern1, pattern2,
                                              interaction_strength)

    def _patterns_overlap(self, pattern1: EmergentPattern,
                          pattern2: EmergentPattern) -> bool:
        """Проверява за припокриване на модели"""
        total_radius = pattern1.influence_radius + pattern2.influence_radius
        return len(set(pattern1.affected_entities) &
                   set(pattern2.affected_entities)) > 0

    def _is_constructive(self, pattern1: EmergentPattern,
                         pattern2: EmergentPattern) -> bool:
        """Определя дали взаимодействието е конструктивно"""
        coherence_product = (pattern1.properties['coherence_factor'] *
                             pattern2.properties['coherence_factor'])
        return coherence_product > 0

    def get_emergence_state(self) -> Dict:
        """Връща текущото състояние на емерджентната система"""
        return {
            'complexity': self.complexity,
            'coherence': self.coherence,
            'entropy': self.entropy,
            'active_patterns': len(self.active_patterns),
            'total_strength': sum(p.strength for p in self.active_patterns),
            'pattern_distribution': self._get_pattern_distribution()
        }

    def _get_pattern_distribution(self) -> Dict[int, float]:
        """Изчислява разпределението на активните модели"""
        distribution = {}
        total_strength = sum(p.strength for p in self.active_patterns)

        if total_strength > 0:
            for pattern in self.active_patterns:
                distribution[pattern.pattern_id] = pattern.strength / \
                    total_strength

        return distribution
