from enum import Enum, auto
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time
import numpy as np

from utils.logger import Logger
from config import GAME_CONFIG


class GameState(Enum):
    MENU = auto()
    CALIBRATION = auto()
    PLAYING = auto()
    PAUSED = auto()
    LEVEL_UP = auto()
    GAME_OVER = auto()
    SETTINGS = auto()
    LEADERBOARD = auto()
    TRANSITION = auto()


@dataclass
class PlayerMetrics:
    """Метрики за представянето на играча"""
    accuracy: float = 0.0
    reaction_time: float = 0.0
    movement_precision: float = 0.0
    target_efficiency: float = 0.0
    score: int = 0
    shots_fired: int = 0
    shots_hit: int = 0
    current_streak: int = 0
    best_streak: int = 0
    session_time: float = 0.0
    last_activity_time: float = 0.0


@dataclass
class PlayerState:
    """Текущо състояние на играча"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    orientation: float
    health: float
    energy: float
    metrics: PlayerMetrics


class GameStateManager:
    """Управлява състоянията на играта и преходите между тях"""

    def __init__(self):
        self.current_state = GameState.MENU
        self.previous_state = None
        self.transition_start = 0.0
        self.transition_duration = 0.5
        self.config = GAME_CONFIG
        self.logger = Logger()

        # Инициализация на играча
        self.player_state = self._initialize_player_state()

        # Състояния и техните handler-и
        self.state_handlers = {
            GameState.MENU: self._handle_menu,
            GameState.CALIBRATION: self._handle_calibration,
            GameState.PLAYING: self._handle_playing,
            GameState.PAUSED: self._handle_paused,
            GameState.LEVEL_UP: self._handle_level_up,
            GameState.GAME_OVER: self._handle_game_over,
            GameState.SETTINGS: self._handle_settings,
            GameState.LEADERBOARD: self._handle_leaderboard,
            GameState.TRANSITION: self._handle_transition
        }

        # История на състоянията за анализ
        self.state_history = []
        self.max_history_length = 1000

        # Callbacks за различни събития
        self.callbacks = {
            'on_state_change': [],
            'on_score_change': [],
            'on_level_up': [],
            'on_game_over': []
        }

    def _initialize_player_state(self) -> PlayerState:
        """Инициализира началното състояние на играча"""
        screen_center = (
            self.config.screen_size[0] / 2,
            self.config.screen_size[1] / 2
        )

        return PlayerState(
            position=screen_center,
            velocity=(0.0, 0.0),
            acceleration=(0.0, 0.0),
            orientation=0.0,
            health=100.0,
            energy=100.0,
            metrics=PlayerMetrics(last_activity_time=time.time())
        )

    def update(self, delta_time: float):
        """Обновява текущото състояние"""
        # Проверка за неактивност
        if self.current_state == GameState.PLAYING:
            self._check_inactivity()

        # Обновяване на метрики
        self.player_state.metrics.session_time += delta_time

        # Изпълнение на handler-а за текущото състояние
        if self.current_state in self.state_handlers:
            self.state_handlers[self.current_state](delta_time)

        # Запазване на историята
        self._update_state_history()

    def transition_to(self, new_state: GameState):
        """Преход към ново състояние"""
        if new_state == self.current_state:
            return

        self.logger.info(f"Transitioning from {
                         self.current_state} to {new_state}")

        self.previous_state = self.current_state
        self.current_state = GameState.TRANSITION
        self.transition_start = time.time()

        # Извикване на callbacks
        self._trigger_callbacks('on_state_change', new_state)

    def handle_event(self, event):
        """Обработка на събития според текущото състояние"""
        if self.current_state == GameState.PLAYING:
            self._handle_playing_event(event)
        elif self.current_state == GameState.MENU:
            self._handle_menu_event(event)
        # ... други състояния

    def register_callback(self, event_type: str, callback):
        """Регистрира callback за определен тип събитие"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)

    def _check_inactivity(self):
        """Проверява за неактивност на играча"""
        current_time = time.time()
        if (current_time - self.player_state.metrics.last_activity_time >
                self.config.inactivity_timeout):
            self.transition_to(GameState.GAME_OVER)

    def _update_state_history(self):
        """Обновява историята на състоянията"""
        self.state_history.append({
            'state': self.current_state,
            'time': time.time(),
            'metrics': self.player_state.metrics
        })

        # Ограничаване на размера на историята
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Извиква регистрираните callbacks за събитие"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in callback {
                                  callback.__name__}: {e}")

    # Handler-и за различните състояния
    def _handle_playing(self, delta_time: float):
        """Обработка на състояние PLAYING"""
        # Обновяване на физиката
        self._update_player_physics(delta_time)

        # Проверка за условия за преход
        if self._check_level_up_conditions():
            self.transition_to(GameState.LEVEL_UP)
        elif self._check_game_over_conditions():
            self.transition_to(GameState.GAME_OVER)

    def _update_player_physics(self, delta_time: float):
        """Обновява физиката на играча"""
        # Обновяване на позицията базирано на скоростта
        self.player_state.position = (
            self.player_state.position[0] +
            self.player_state.velocity[0] * delta_time,
            self.player_state.position[1] +
            self.player_state.velocity[1] * delta_time
        )

        # Обновяване на скоростта базирано на ускорението
        self.player_state.velocity = (
            self.player_state.velocity[0] +
            self.player_state.acceleration[0] * delta_time,
            self.player_state.velocity[1] +
            self.player_state.acceleration[1] * delta_time
        )

    def get_player_state(self) -> PlayerState:
        """Връща текущото състояние на играча"""
        return self.player_state

    def get_player_metrics(self) -> PlayerMetrics:
        """Връща текущите метрики на играча"""
        return self.player_state.metrics
