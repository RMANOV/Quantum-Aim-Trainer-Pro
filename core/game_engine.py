import pygame
import numpy as np
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass

from core.state_manager import GameStateManager
from core.physics_engine import PhysicsEngine
from neural.predictor import NeuralPredictionNetwork
from quantum.field_simulator import QuantumFieldSimulator
from graphics.renderer import GameRenderer
from graphics.particle_system import ParticleSystem
from graphics.effects_processor import VisualEffectsProcessor
from gameplay.challenge_generator import DynamicChallengeGenerator
from gameplay.target_system import TargetSystem
from gameplay.difficulty_manager import DifficultyManager
from utils.logger import Logger
from utils.profiler import Profiler
from config import GAME_CONFIG


@dataclass
class GameMetrics:
    fps: float = 0.0
    frame_time: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    memory_usage: float = 0.0
    prediction_accuracy: float = 0.0
    quantum_coherence: float = 0.0


class GameEngine:
    """Основной игровой движок, координирующий все подсистемы"""

    def __init__(self):
        pygame.init()
        self.config = GAME_CONFIG
        self.screen = self._initialize_display()
        self.clock = pygame.time.Clock()
        self.running = False

        # Инициализация всех подсистем
        self.state_manager = GameStateManager()
        self.physics_engine = PhysicsEngine()
        self.neural_predictor = NeuralPredictionNetwork()
        self.quantum_field = QuantumFieldSimulator(self.config.screen_size)
        self.renderer = GameRenderer(self.screen)
        self.particle_system = ParticleSystem()
        self.effects_processor = VisualEffectsProcessor(
            self.config.screen_size)
        self.challenge_generator = DynamicChallengeGenerator(
            self.neural_predictor)
        self.target_system = TargetSystem()
        self.difficulty_manager = DifficultyManager()

        # Утилиты
        self.logger = Logger()
        self.profiler = Profiler()

        # Метрики
        self.metrics = GameMetrics()

    def _initialize_display(self) -> pygame.Surface:
        """Инициализация дисплея с поддержкой высокой частоты обновления"""
        pygame.display.set_caption("Quantum Aim Trainer Pro")

        # Получаем информацию о доступных режимах дисплея
        display_modes = pygame.display.list_modes()

        if self.config.fullscreen:
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            if self.config.vsync:
                flags |= pygame.VSYNC

            screen = pygame.display.set_mode(
                self.config.screen_size,
                flags
            )
        else:
            screen = pygame.display.set_mode(
                self.config.screen_size,
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )

        # Установка частоты обновления
        if hasattr(pygame.display, 'set_refresh_rate'):
            try:
                pygame.display.set_refresh_rate(144)
            except:
                self.logger.warning("Could not set high refresh rate")

        return screen

    def run(self):
        """Основной игровой цикл"""
        self.running = True
        last_time = time.time()

        try:
            while self.running:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time

                with self.profiler.measure("frame"):
                    self._process_frame(delta_time)

                # Обновление метрик
                self._update_metrics()

                # Поддержание частоты кадров
                self.clock.tick(self.config.target_fps)

        except Exception as e:
            self.logger.error(f"Critical error in game loop: {e}")
            raise
        finally:
            self.cleanup()

    def _process_frame(self, delta_time: float):
        """Обработка одного кадра"""
        # Обработка событий
        self._handle_events()

        # Обновление состояния игры
        with self.profiler.measure("update"):
            self._update(delta_time)

        # Рендеринг
        with self.profiler.measure("render"):
            self._render()

    def _handle_events(self):
        """Обработка событий pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state_manager.toggle_pause()

            # Передача событий в менеджер состояний
            self.state_manager.handle_event(event)

    def _update(self, delta_time: float):
        """Обновление всех подсистем"""
        # Обновление квантового поля
        self.quantum_field.evolve(delta_time)

        # Обновление физики
        self.physics_engine.update(delta_time)

        # Обновление нейронного предсказателя
        player_state = self.state_manager.get_player_state()
        prediction = self.neural_predictor.process_player_action(player_state)

        # Обновление целей
        self.target_system.update(delta_time, prediction)

        # Обновление частиц
        self.particle_system.update(delta_time)

        # Обновление сложности
        self.difficulty_manager.update(self.state_manager.get_player_metrics())

    def _render(self):
        """Рендеринг кадра"""
        # Очистка экрана
        self.screen.fill((0, 0, 20))  # Тёмно-синий фон

        # Рендеринг квантового поля
        quantum_surface = self.quantum_field.render()
        self.screen.blit(quantum_surface, (0, 0))

        # Рендеринг игровых объектов
        self.renderer.render_game_objects(
            self.target_system.get_targets(),
            self.particle_system.get_particles()
        )

        # Применение визуальных эффектов
        self.screen = self.effects_processor.process_frame(self.screen)

        # Обновление экрана
        pygame.display.flip()

    def _update_metrics(self):
        """Обновление метрик производительности"""
        self.metrics.fps = self.clock.get_fps()
        self.metrics.frame_time = self.profiler.get_last_frame_time()
        self.metrics.prediction_accuracy = self.neural_predictor.get_accuracy()
        self.metrics.quantum_coherence = self.quantum_field.get_coherence()

    def cleanup(self):
        """Очистка ресурсов при завершении"""
        self.logger.info("Cleaning up resources...")
        pygame.quit()
        self.neural_predictor.save_model()
        self.logger.close()
