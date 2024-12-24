import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy import stats
from collections import deque
import time

from utils.logger import Logger
from config import ANALYZER_CONFIG


@dataclass
class BehaviorMetrics:
    """Метрики за анализ на поведението на играча"""
    accuracy_trend: float = 0.0
    reaction_time_trend: float = 0.0
    movement_patterns: List[str] = None
    skill_rating: float = 0.0
    learning_rate: float = 0.0
    consistency_score: float = 0.0
    adaptation_speed: float = 0.0
    fatigue_index: float = 0.0

    def __post_init__(self):
        if self.movement_patterns is None:
            self.movement_patterns = []


class BehaviorAnalyzer:
    """Анализатор на поведението на играча с машинно обучение"""

    def __init__(self):
        self.config = ANALYZER_CONFIG
        self.logger = Logger()

        # Буфери за данни
        self.movement_buffer = deque(maxlen=1000)
        self.accuracy_buffer = deque(maxlen=100)
        self.reaction_times = deque(maxlen=100)
        self.session_data = []

        # Невронна мрежа за анализ на модели
        self.pattern_analyzer = self._build_pattern_analyzer()

        # Текущи метрики
        self.metrics = BehaviorMetrics()

        # Времеви променливи
        self.session_start = time.time()
        self.last_update = time.time()

    def _build_pattern_analyzer(self) -> nn.Module:
        """Създава невронна мрежа за анализ на поведенчески модели"""
        class PatternNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(ANALYZER_CONFIG.input_size, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),

                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),

                    nn.Linear(64, 32),
                    nn.ReLU()
                )

                self.pattern_classifier = nn.Linear(
                    32, ANALYZER_CONFIG.num_patterns)

            def forward(self, x):
                features = self.encoder(x)
                patterns = self.pattern_classifier(features)
                return patterns, features

        return PatternNetwork()

    def update(self, player_state: Dict, action_result: Dict):
        """Обновява анализа с нови данни"""
        current_time = time.time()

        # Запазване на данните
        self._store_data(player_state, action_result, current_time)

        # Анализ на поведението
        self._analyze_behavior()

        # Обновяване на метрики
        self._update_metrics()

        self.last_update = current_time

    def _store_data(self, player_state: Dict, action_result: Dict, timestamp: float):
        """Съхранява данните за анализ"""
        # Съхраняване на движения
        if 'position' in player_state:
            self.movement_buffer.append({
                'position': player_state['position'],
                'velocity': player_state.get('velocity', (0, 0)),
                'timestamp': timestamp
            })

        # Съхраняване на точност
        if 'hit' in action_result:
            self.accuracy_buffer.append(action_result['hit'])

        # Съхраняване на време за реакция
        if 'reaction_time' in action_result:
            self.reaction_times.append(action_result['reaction_time'])

        # Съхраняване на сесийни данни
        self.session_data.append({
            'state': player_state,
            'result': action_result,
            'timestamp': timestamp
        })

    def _analyze_behavior(self):
        """Анализира поведението на играча"""
        self._analyze_movement_patterns()
        self._analyze_performance_trends()
        self._analyze_skill_progression()
        self._analyze_fatigue()

    def _analyze_movement_patterns(self):
        """Анализира моделите на движение"""
        if len(self.movement_buffer) < self.config.min_pattern_length:
            return

        # Подготовка на данните за анализ
        movement_data = np.array([
            [m['position'][0], m['position'][1],
             m['velocity'][0], m['velocity'][1]]
            for m in self.movement_buffer
        ])

        # Нормализация
        movement_data = (movement_data - np.mean(movement_data, axis=0)) / \
            (np.std(movement_data, axis=0) + 1e-7)

        # Анализ чрез невронната мрежа
        with torch.no_grad():
            data_tensor = torch.FloatTensor(movement_data)
            patterns, features = self.pattern_analyzer(data_tensor)

            # Идентифициране на доминиращи модели
            pattern_probs = torch.softmax(patterns, dim=-1)
            dominant_patterns = torch.argsort(
                pattern_probs, descending=True)[:3]

            self.metrics.movement_patterns = [
                self.config.pattern_names[i] for i in dominant_patterns
            ]

    def _analyze_performance_trends(self):
        """Анализира тенденциите в представянето"""
        if len(self.accuracy_buffer) > 10:
            # Анализ на точността
            accuracy_trend = self._calculate_trend(list(self.accuracy_buffer))
            self.metrics.accuracy_trend = accuracy_trend

        if len(self.reaction_times) > 10:
            # Анализ на времето за реакция
            reaction_trend = self._calculate_trend(list(self.reaction_times))
            self.metrics.reaction_time_trend = -reaction_trend  # Обратна връзка

    def _analyze_skill_progression(self):
        """Анализира прогреса на уменията"""
        if len(self.session_data) < 10:
            return

        # Изчисляване на базово ниво на умения
        recent_accuracy = np.mean(list(self.accuracy_buffer))
        recent_reaction = np.mean(list(self.reaction_times))

        # Комбиниран скор за умения
        skill_score = (recent_accuracy * 0.7 +
                       (1.0 / (recent_reaction + 1e-5)) * 0.3)

        # Изчисляване на скорост на учене
        if len(self.session_data) > 100:
            early_skill = self._calculate_skill_score(self.session_data[:50])
            latest_skill = self._calculate_skill_score(self.session_data[-50:])

            self.metrics.learning_rate = (
                latest_skill - early_skill) / early_skill

        self.metrics.skill_rating = skill_score

    def _analyze_fatigue(self):
        """Анализира умората на играча"""
        if len(self.reaction_times) < 20:
            return

        # Анализ на вариацията във времето за реакция
        reaction_std = np.std(list(self.reaction_times))
        baseline_std = np.std(list(self.reaction_times)[:20])

        fatigue_index = (reaction_std / baseline_std) - 1.0
        self.metrics.fatigue_index = max(0, fatigue_index)

    def _calculate_trend(self, data: List[float]) -> float:
        """Изчислява тренд в данните"""
        if not data:
            return 0.0

        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)

        # Нормализиран тренд
        return slope * r_value**2

    def _calculate_skill_score(self, data: List[Dict]) -> float:
        """Изчислява скор на уменията за даден период"""
        accuracies = [d['result'].get('hit', 0)
                      for d in data if 'hit' in d['result']]
        reactions = [d['result'].get('reaction_time', 1.0)
                     for d in data if 'reaction_time' in d['result']]

        if not accuracies or not reactions:
            return 0.0

        avg_accuracy = np.mean(accuracies)
        avg_reaction = np.mean(reactions)

        return (avg_accuracy * 0.7 + (1.0 / (avg_reaction + 1e-5)) * 0.3)

    def get_analysis(self) -> Dict:
        """Връща пълен анализ на поведението"""
        return {
            'metrics': self.metrics,
            'session_duration': time.time() - self.session_start,
            'dominant_patterns': self.metrics.movement_patterns[:3],
            'skill_assessment': {
                'current_rating': self.metrics.skill_rating,
                'learning_rate': self.metrics.learning_rate,
                'consistency': self.metrics.consistency_score,
                'fatigue_level': self.metrics.fatigue_index
            },
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Генерира препоръки базирани на анализа"""
        recommendations = []

        if self.metrics.fatigue_index > 0.3:
            recommendations.append(
                "Consider taking a short break to maintain performance")

        if self.metrics.accuracy_trend < 0:
            recommendations.append("Focus on accuracy over speed")

        if self.metrics.learning_rate < 0.1:
            recommendations.append("Try varying your practice patterns")

        return recommendations
