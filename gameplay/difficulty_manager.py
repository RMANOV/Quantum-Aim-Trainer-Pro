import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import math
from scipy import stats

from utils.logger import Logger
from config import DIFFICULTY_CONFIG


@dataclass
class DifficultyMetrics:
    """Метрики за трудността"""
    current_level: int
    base_difficulty: float
    dynamic_multiplier: float
    adaptation_rate: float
    player_skill_rating: float
    challenge_factor: float
    stress_level: float
    engagement_score: float


class DifficultyManager:
    """Мениджър за адаптивна трудност с машинно обучение"""

    def __init__(self):
        self.config = DIFFICULTY_CONFIG
        self.logger = Logger()

        # Основни параметри
        self.current_level = 1
        self.base_difficulty = 1.0
        self.dynamic_multiplier = 1.0

        # История и анализ
        self.performance_history = []
        self.adaptation_history = []
        self.skill_curve = []

        # Времеви параметри
        self.session_start = time.time()
        self.last_adjustment = time.time()

        # Метрики
        self.metrics = DifficultyMetrics(
            current_level=1,
            base_difficulty=1.0,
            dynamic_multiplier=1.0,
            adaptation_rate=0.1,
            player_skill_rating=0.5,
            challenge_factor=1.0,
            stress_level=0.0,
            engagement_score=1.0
        )

        # Невронни тежести за различни фактори
        self.factor_weights = {
            'accuracy': 0.3,
            'reaction_time': 0.2,
            'movement_precision': 0.2,
            'target_efficiency': 0.15,
            'stress_resistance': 0.15
        }

    def update(self, player_metrics: Dict):
        """Обновява трудността базирано на представянето на играча"""
        current_time = time.time()

        # Анализ на представянето
        performance_score = self._analyze_performance(player_metrics)
        self.performance_history.append(performance_score)

        # Обновяване на уменията на играча
        self._update_skill_rating(performance_score)

        # Адаптивна корекция на трудността
        if current_time - self.last_adjustment > self.config.adjustment_interval:
            self._adjust_difficulty()
            self.last_adjustment = current_time

        # Обновяване на метриките
        self._update_metrics(player_metrics)

    def _analyze_performance(self, metrics: Dict) -> float:
        """Анализира представянето на играча"""
        score = 0.0

        for factor, weight in self.factor_weights.items():
            if factor in metrics:
                factor_score = self._normalize_metric(metrics[factor], factor)
                score += factor_score * weight

        # Добавяне на нелинейни фактори
        score *= (1.0 + self.metrics.engagement_score * 0.2)
        score *= (1.0 - self.metrics.stress_level * 0.3)

        return np.clip(score, 0.0, 1.0)

    def _normalize_metric(self, value: float, metric_type: str) -> float:
        """Нормализира метрика в интервала [0, 1]"""
        if metric_type == 'accuracy':
            return value  # Вече е в [0, 1]
        elif metric_type == 'reaction_time':
            # По-бързо време = по-висок скор
            return 1.0 / (1.0 + value / 500.0)  # 500ms като базово време
        elif metric_type == 'movement_precision':
            return np.clip(value / 100.0, 0.0, 1.0)
        else:
            return np.clip(value, 0.0, 1.0)

    def _update_skill_rating(self, performance_score: float):
        """Обновява рейтинга на уменията на играча"""
        # Експоненциално плъзгащо средно
        alpha = 0.1
        self.metrics.player_skill_rating = (alpha * performance_score +
                                            (1 - alpha) * self.metrics.player_skill_rating)

        self.skill_curve.append(self.metrics.player_skill_rating)

    def _adjust_difficulty(self):
        """Коригира трудността базирано на представянето"""
        if len(self.performance_history) < 2:
            return

        # Анализ на тренда в представянето
        recent_performance = self.performance_history[-10:]
        performance_trend = self._calculate_trend(recent_performance)

        # Изчисляване на оптималната трудност
        target_difficulty = self._calculate_optimal_difficulty()

        # Плавна адаптация
        adaptation_rate = self._calculate_adaptation_rate()
        self.dynamic_multiplier += (target_difficulty -
                                    self.dynamic_multiplier) * adaptation_rate

        # Запазване на адаптацията
        self.adaptation_history.append({
            'time': time.time(),
            'multiplier': self.dynamic_multiplier,
            'performance': np.mean(recent_performance),
            'trend': performance_trend
        })

    def _calculate_trend(self, data: List[float]) -> float:
        """Изчислява тренд в данните"""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)

        return slope * r_value**2

    def _calculate_optimal_difficulty(self) -> float:
        """Изчислява оптимална трудност за текущото ниво на умения"""
        skill_rating = self.metrics.player_skill_rating
        engagement = self.metrics.engagement_score

        # Базова трудност базирана на уменията
        optimal = 0.5 + skill_rating * 0.5

        # Корекция базирана на ангажираността
        optimal *= 1.0 + (engagement - 0.5) * 0.2

        # Добавяне на случаен фактор за непредвидимост
        optimal *= 1.0 + np.random.normal(0, 0.1)

        return np.clip(optimal, 0.5, 2.0)

    def _calculate_adaptation_rate(self) -> float:
        """Изчислява скоростта на адаптация"""
        # По-бърза адаптация при големи разлики в представянето
        performance_variance = np.std(self.performance_history[-10:])
        base_rate = self.metrics.adaptation_rate

        return base_rate * (1.0 + performance_variance)

    def _update_metrics(self, player_metrics: Dict):
        """Обновява метриките на системата"""
        # Обновяване на стрес нивото
        self.metrics.stress_level = self._calculate_stress_level(
            player_metrics)

        # Обновяване на ангажираността
        self.metrics.engagement_score = self._calculate_engagement(
            player_metrics)

        # Обновяване на фактора на предизвикателство
        self.metrics.challenge_factor = self._calculate_challenge_factor()

    def get_current_parameters(self) -> Dict:
        """Връща текущите параметри на трудността"""
        effective_difficulty = self.base_difficulty * self.dynamic_multiplier

        return {
            'target_speed': self.config.base_speed * effective_difficulty,
            'target_size': self.config.base_size / effective_difficulty,
            'spawn_rate': self.config.base_spawn_rate * effective_difficulty,
            'fake_target_ratio': min(0.3, 0.1 * effective_difficulty),
            'pattern_complexity': min(1.0, 0.5 * effective_difficulty),
            'quantum_influence': min(0.8, 0.2 * effective_difficulty)
        }
