import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
from dataclasses import dataclass, asdict
import logging
from copy import deepcopy


@dataclass
class GraphicsConfig:
    screen_width: int = 1920
    screen_height: int = 1080
    fps_limit: int = 144
    vsync: bool = True
    bloom_intensity: float = 0.5
    particle_limit: int = 10000
    shader_quality: str = "high"
    post_processing: bool = True
    antialiasing: str = "MSAA_4x"


@dataclass
class PhysicsConfig:
    gravity: float = 9.81
    air_resistance: float = 0.02
    collision_iterations: int = 8
    time_step: float = 1/240
    velocity_iterations: int = 8
    position_iterations: int = 3


@dataclass
class GameplayConfig:
    difficulty_scaling: float = 1.0
    target_spawn_rate: float = 1.0
    score_multiplier: float = 1.0
    quantum_effect_strength: float = 0.5
    pattern_complexity: float = 1.0
    adaptation_rate: float = 0.1


class ConfigManager:
    """Менаджър за конфигурации с поддръжка на профили и валидация"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Текущи конфигурации
        self.graphics = GraphicsConfig()
        self.physics = PhysicsConfig()
        self.gameplay = GameplayConfig()

        # Кеширани конфигурации
        self.config_cache: Dict[str, Dict] = {}

        # Валидатори за различните секции
        self.validators = {
            'graphics': self._validate_graphics,
            'physics': self._validate_physics,
            'gameplay': self._validate_gameplay
        }

        # Зареждане на конфигурациите
        self.load_configs()

    def load_configs(self):
        """Зарежда всички конфигурационни файлове"""
        # Основна конфигурация
        self._load_base_config()

        # Потребителски настройки
        self._load_user_config()

        # Профилни конфигурации
        self._load_profiles()

        # Валидация на заредените конфигурации
        self._validate_all_configs()

    def _load_base_config(self):
        """Зарежда базовата конфигурация"""
        base_config_path = self.config_dir / "base_config.yaml"

        if base_config_path.exists():
            with open(base_config_path) as f:
                base_config = yaml.safe_load(f)
                self._apply_config(base_config)

    def _load_user_config(self):
        """Зарежда потребителските настройки"""
        user_config_path = self.config_dir / "user_config.yaml"

        if user_config_path.exists():
            with open(user_config_path) as f:
                user_config = yaml.safe_load(f)
                self._apply_config(user_config, override=True)

    def _load_profiles(self):
        """Зарежда профилните конфигурации"""
        profiles_dir = self.config_dir / "profiles"
        if not profiles_dir.exists():
            return

        for profile_file in profiles_dir.glob("*.yaml"):
            try:
                with open(profile_file) as f:
                    profile_config = yaml.safe_load(f)
                    self.config_cache[profile_file.stem] = profile_config
            except Exception as e:
                logging.error(f"Error loading profile {profile_file}: {e}")

    def _apply_config(self, config: Dict, override: bool = False):
        """Прилага конфигурация към текущите настройки"""
        if 'graphics' in config:
            self._apply_section(config['graphics'], self.graphics, override)
        if 'physics' in config:
            self._apply_section(config['physics'], self.physics, override)
        if 'gameplay' in config:
            self._apply_section(config['gameplay'], self.gameplay, override)

    def _apply_section(self, config: Dict, target: Any, override: bool):
        """Прилага секция от конфигурацията"""
        for key, value in config.items():
            if hasattr(target, key):
                if override or getattr(target, key) is None:
                    setattr(target, key, value)

    def _validate_graphics(self, config: GraphicsConfig) -> bool:
        """Валидира графичните настройки"""
        valid = True

        # Проверка на резолюцията
        if config.screen_width < 640 or config.screen_height < 480:
            logging.warning("Screen resolution too low, using minimum values")
            config.screen_width = max(640, config.screen_width)
            config.screen_height = max(480, config.screen_height)
            valid = False

        # Проверка на FPS лимита
        if config.fps_limit < 30:
            logging.warning("FPS limit too low, setting to 30")
            config.fps_limit = 30
            valid = False

        # Валидация на качеството на шейдърите
        valid_qualities = ["low", "medium", "high", "ultra"]
        if config.shader_quality not in valid_qualities:
            logging.warning(f"Invalid shader quality, using 'medium'")
            config.shader_quality = "medium"
            valid = False

        return valid

    def _validate_physics(self, config: PhysicsConfig) -> bool:
        """Валидира физичните настройки"""
        valid = True

        # Проверка на гравитацията
        if config.gravity < 0:
            logging.warning(
                "Negative gravity not allowed, using absolute value")
            config.gravity = abs(config.gravity)
            valid = False

        # Проверка на времевата стъпка
        if config.time_step <= 0 or config.time_step > 1/30:
            logging.warning("Invalid time step, using 1/240")
            config.time_step = 1/240
            valid = False

        return valid

    def _validate_gameplay(self, config: GameplayConfig) -> bool:
        """Валидира геймплей настройките"""
        valid = True

        # Проверка на скалирането на трудността
        if config.difficulty_scaling <= 0:
            logging.warning("Invalid difficulty scaling, using 1.0")
            config.difficulty_scaling = 1.0
            valid = False

        # Проверка на множителя на точките
        if config.score_multiplier < 0:
            logging.warning(
                "Negative score multiplier not allowed, using absolute value")
            config.score_multiplier = abs(config.score_multiplier)
            valid = False

        return valid

    def _validate_all_configs(self) -> bool:
        """Валидира всички текущи конфигурации"""
        valid = True

        for section, validator in self.validators.items():
            config = getattr(self, section)
            if not validator(config):
                valid = False

        return valid

    def save_config(self, profile_name: Optional[str] = None):
        """Запазва текущата конфигурация"""
        config = {
            'graphics': asdict(self.graphics),
            'physics': asdict(self.physics),
            'gameplay': asdict(self.gameplay)
        }

        if profile_name:
            # Запазване като профил
            profile_path = self.config_dir / \
                "profiles" / f"{profile_name}.yaml"
            profile_path.parent.mkdir(exist_ok=True)

            with open(profile_path, 'w') as f:
                yaml.dump(config, f)
            self.config_cache[profile_name] = deepcopy(config)
        else:
            # Запазване като потребителска конфигурация
            user_config_path = self.config_dir / "user_config.yaml"
            with open(user_config_path, 'w') as f:
                yaml.dump(config, f)

    def load_profile(self, profile_name: str) -> bool:
        """Зарежда профил"""
        if profile_name in self.config_cache:
            self._apply_config(self.config_cache[profile_name], override=True)
            return self._validate_all_configs()
        return False

    def get_config_summary(self) -> Dict:
        """Връща обобщение на текущата конфигурация"""
        return {
            'graphics': asdict(self.graphics),
            'physics': asdict(self.physics),
            'gameplay': asdict(self.gameplay)
        }
