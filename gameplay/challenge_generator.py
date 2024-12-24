import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import time
from scipy.spatial import distance

from utils.logger import Logger
from config import CHALLENGE_CONFIG


@dataclass
class Challenge:
    """Структура за предизвикателство"""
    challenge_type: str
    difficulty: float
    duration: float
    targets: List[Dict]
    requirements: Dict
    reward: int
    special_effects: List[str]
    quantum_parameters: Optional[Dict] = None


class ChallengeGenerator:
    """Генератор на динамични предизвикателства"""

    def __init__(self, neural_predictor=None):
        self.config = CHALLENGE_CONFIG
        self.logger = Logger()
        self.neural_predictor = neural_predictor

        # Типове предизвикателства
        self.challenge_types = {
            'precision': self._generate_precision_challenge,
            'speed': self._generate_speed_challenge,
            'tracking': self._generate_tracking_challenge,
            'pattern': self._generate_pattern_challenge,
            'quantum': self._generate_quantum_challenge,
            'multi_target': self._generate_multi_target_challenge,
            'adaptive': self._generate_adaptive_challenge
        }

        # История на предизвикателствата
        self.challenge_history = []
        self.player_performance = {}

        # Метрики
        self.difficulty_curve = []
        self.success_rate = []

    def generate_challenge(self, player_metrics: Dict) -> Challenge:
        """Генерира ново предизвикателство базирано на метриките на играча"""
        # Анализ на слабостите на играча
        weaknesses = self._analyze_weaknesses(player_metrics)

        # Избор на тип предизвикателство
        challenge_type = self._select_challenge_type(weaknesses)

        # Изчисляване на трудността
        difficulty = self._calculate_difficulty(player_metrics)

        # Генериране на предизвикателството
        generator = self.challenge_types.get(challenge_type,
                                             self.challenge_types['precision'])
        challenge = generator(difficulty, player_metrics)

        # Запазване в историята
        self.challenge_history.append({
            'type': challenge_type,
            'difficulty': difficulty,
            'timestamp': time.time()
        })

        return challenge

    def _analyze_weaknesses(self, metrics: Dict) -> Dict[str, float]:
        """Анализира слабите страни на играча"""
        weaknesses = {}

        # Анализ на точността
        if 'accuracy' in metrics:
            weaknesses['precision'] = 1.0 - metrics['accuracy']

        # Анализ на скоростта
        if 'reaction_time' in metrics:
            weaknesses['speed'] = min(1.0, metrics['reaction_time'] / 1000.0)

        # Анализ на проследяването
        if 'tracking_accuracy' in metrics:
            weaknesses['tracking'] = 1.0 - metrics['tracking_accuracy']

        # Анализ на разпознаването на модели
        if 'pattern_recognition' in metrics:
            weaknesses['pattern'] = 1.0 - metrics['pattern_recognition']

        return weaknesses

    def _select_challenge_type(self, weaknesses: Dict[str, float]) -> str:
        """Избира тип предизвикателство базирано на слабостите"""
        if not weaknesses:
            return random.choice(list(self.challenge_types.keys()))

        # Вероятностен избор базиран на слабостите
        total = sum(weaknesses.values())
        if total == 0:
            return random.choice(list(self.challenge_types.keys()))

        r = random.uniform(0, total)
        cumsum = 0
        for type_, weight in weaknesses.items():
            cumsum += weight
            if r <= cumsum:
                return type_

        return random.choice(list(self.challenge_types.keys()))

    def _calculate_difficulty(self, metrics: Dict) -> float:
        """Изчислява подходяща трудност"""
        # Базова трудност от уменията на играча
        if 'skill_rating' in metrics:
            base_difficulty = metrics['skill_rating']
        else:
            base_difficulty = 0.5

        # Корекция базирана на историята на успеха
        if self.success_rate:
            success_adjustment = (np.mean(self.success_rate) - 0.7) * 0.3
            base_difficulty += success_adjustment

        # Добавяне на прогресивно увеличение
        progression = len(self.challenge_history) * 0.01

        # Добавяне на случаен фактор
        randomness = random.uniform(-0.1, 0.1)

        return np.clip(base_difficulty + progression + randomness, 0.1, 1.0)

    def _generate_precision_challenge(self, difficulty: float,
                                      metrics: Dict) -> Challenge:
        """Генерира предизвикателство за точност"""
        return Challenge(
            challenge_type='precision',
            difficulty=difficulty,
            duration=30.0,
            targets=self._generate_precision_targets(difficulty),
            requirements={
                'accuracy': 0.7 + difficulty * 0.2,
                'min_hits': int(10 + difficulty * 20)
            },
            reward=int(1000 * difficulty),
            special_effects=['precision_guide', 'target_highlight']
        )

    def _generate_quantum_challenge(self, difficulty: float,
                                    metrics: Dict) -> Challenge:
        """Генерира квантово предизвикателство"""
        quantum_params = {
            'superposition_factor': difficulty * 0.5,
            'entanglement_strength': difficulty * 0.3,
            'uncertainty_radius': 20 + difficulty * 30,
            'phase_shift_rate': 0.5 + difficulty * 1.5
        }

        return Challenge(
            challenge_type='quantum',
            difficulty=difficulty,
            duration=25.0,
            targets=self._generate_quantum_targets(difficulty, quantum_params),
            requirements={
                'quantum_accuracy': 0.6 + difficulty * 0.2,
                'phase_synchronization': 0.5 + difficulty * 0.3
            },
            reward=int(1500 * difficulty),
            special_effects=['quantum_trail',
                             'phase_indicator', 'uncertainty_field'],
            quantum_parameters=quantum_params
        )

    def _generate_quantum_targets(self, difficulty: float,
                                  quantum_params: Dict) -> List[Dict]:
        """Генерира мишени с квантово поведение"""
        targets = []
        num_targets = int(3 + difficulty * 5)

        for i in range(num_targets):
            phase = random.uniform(0, 2 * np.pi)
            position = np.random.rand(2) * self.config.screen_size

            target = {
                'position': position,
                'phase': phase,
                'superposition_states': self._generate_superposition_states(
                    position, quantum_params),
                'entanglement_pairs': [],
                'quantum_behavior': {
                    'uncertainty': quantum_params['uncertainty_radius'],
                    'phase_shift': quantum_params['phase_shift_rate'],
                    'collapse_probability': difficulty * 0.2
                }
            }

            targets.append(target)

        # Създаване на заплетени двойки
        self._create_entangled_pairs(targets, difficulty)

        return targets

    def _generate_superposition_states(self, base_position: np.ndarray,
                                       quantum_params: Dict) -> List[Dict]:
        """Генерира състояния на суперпозиция за мишена"""
        states = []
        num_states = int(2 + quantum_params['superposition_factor'] * 3)

        for _ in range(num_states):
            offset = np.random.normal(
                0, quantum_params['uncertainty_radius'], 2)
            probability = random.uniform(0.2, 1.0)

            states.append({
                'position': base_position + offset,
                'probability': probability,
                'phase': random.uniform(0, 2 * np.pi)
            })

        return states

    def _create_entangled_pairs(self, targets: List[Dict], difficulty: float):
        """Създава заплетени двойки от мишени"""
        num_pairs = int(len(targets) * difficulty * 0.5)

        for _ in range(num_pairs):
            if len(targets) < 2:
                break

            # Избор на двойка мишени
            idx1, idx2 = random.sample(range(len(targets)), 2)

            # Създаване на заплитане
            entanglement = {
                'strength': random.uniform(0.5, 1.0),
                'type': random.choice(['position', 'phase', 'combined'])
            }

            targets[idx1]['entanglement_pairs'].append((idx2, entanglement))
            targets[idx2]['entanglement_pairs'].append((idx1, entanglement))
