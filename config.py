from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class GAME_CONFIG:
    """Основни настройки на играта"""
    # Екран и графика
    screen_size: Tuple[int, int] = (1920, 1080)
    fullscreen: bool = True
    vsync: bool = True
    target_fps: int = 144

    # Производителност
    max_particles: int = 10000
    physics_iterations: int = 8
    max_targets: int = 100

    # Игрови параметри
    inactivity_timeout: float = 420.0  # 7 минути
    min_reaction_time: float = 0.1  # seconds
    base_target_lifetime: float = 2.0  # seconds


@dataclass
class GRAPHICS_CONFIG:
    """Графични настройки"""
    # Пост-процесинг
    bloom_enabled: bool = True
    bloom_intensity: float = 0.5
    chromatic_aberration: float = 0.2
    motion_blur_strength: float = 0.1

    # Частици
    particle_quality: str = "high"
    max_particle_systems: int = 5
    particle_lifetime: float = 2.0

    # Визуални ефекти
    glow_intensity: float = 0.8
    trail_length: int = 20
    shake_intensity: float = 0.5


@dataclass
class PHYSICS_CONFIG:
    """Физични настройки"""
    gravity: Tuple[float, float] = (0.0, 9.81)
    damping: float = 0.99
    air_resistance: float = 0.02
    collision_threshold: float = 0.01

    # Времеви параметри
    min_step: float = 1/240
    max_step: float = 1/60


@dataclass
class NEURAL_CONFIG:
    """Настройки за невронната мрежа"""
    sequence_length: int = 100
    feature_dimension: int = 8
    hidden_units: Tuple[int, ...] = (64, 32, 16)
    learning_rate: float = 0.001
    batch_size: int = 32
    prediction_threshold: float = 0.7


@dataclass
class QUANTUM_CONFIG:
    """Настройки за квантовата симулация"""
    state_size: int = 16
    coherence_threshold: float = 0.7
    entanglement_strength: float = 0.5
    uncertainty_radius: float = 30.0

    # Полева симулация
    cell_size: int = 20
    potential_strength: float = 1.0
    noise_strength: float = 0.1

    # Прагове
    vortex_threshold: float = 0.5
    interference_threshold: float = 0.3


@dataclass
class ANALYZER_CONFIG:
    """Настройки за анализатора"""
    input_size: int = 64
    hidden_size: int = 128
    num_patterns: int = 8
    min_pattern_length: int = 10

    # Анализ
    performance_window: int = 100
    skill_decay_rate: float = 0.01
    adaptation_rate: float = 0.1


@dataclass
class DIFFICULTY_CONFIG:
    """Настройки за трудността"""
    base_speed: float = 100.0
    base_size: float = 30.0
    base_spawn_rate: float = 1.0

    # Адаптация
    adjustment_interval: float = 5.0
    min_difficulty: float = 0.5
    max_difficulty: float = 2.0
    progression_rate: float = 0.1


@dataclass
class TARGET_CONFIG:
    """Настройки за мишените"""
    screen_width: int = 1920
    screen_height: int = 1080
    base_size: float = 30.0
    min_lifetime: float = 0.5
    max_lifetime: float = 5.0

    # Движение
    max_speed: float = 500.0
    acceleration: float = 200.0
    rotation_speed: float = 2.0


@dataclass
class PARTICLE_CONFIG:
    """Настройки за частиците"""
    max_particles: int = 10000
    emission_rate: float = 100.0
    base_lifetime: float = 2.0

    # Физика на частиците
    min_speed: float = 50.0
    max_speed: float = 200.0
    fade_rate: float = 0.5

    # Визуални параметри
    min_size: float = 2.0
    max_size: float = 8.0
    glow_strength: float = 0.5


@dataclass
class EFFECTS_CONFIG:
    """Настройки за визуалните ефекти"""
    bloom_radius: int = 15
    bloom_intensity: float = 0.5
    chromatic_strength: float = 0.2
    wave_frequency: float = 2.0
    wave_amplitude: float = 10.0

    # Пост-процесинг
    blur_radius: int = 5
    vignette_strength: float = 0.3
    color_correction: Dict[str, float] = None

    def __post_init__(self):
        if self.color_correction is None:
            self.color_correction = {
                'contrast': 1.1,
                'saturation': 1.2,
                'brightness': 1.0
            }


# Експортиране на конфигурациите
CONFIGS = {
    'game': GAME_CONFIG(),
    'graphics': GRAPHICS_CONFIG(),
    'physics': PHYSICS_CONFIG(),
    'neural': NEURAL_CONFIG(),
    'quantum': QUANTUM_CONFIG(),
    'analyzer': ANALYZER_CONFIG(),
    'difficulty': DIFFICULTY_CONFIG(),
    'target': TARGET_CONFIG(),
    'particle': PARTICLE_CONFIG(),
    'effects': EFFECTS_CONFIG()
}
