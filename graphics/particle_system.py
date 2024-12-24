import pygame
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import colorsys
import time

from utils.logger import Logger
from config import PARTICLE_CONFIG


@dataclass
class Particle:
    """Структура за частица"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    color: Tuple[int, int, int, int]
    size: float
    lifetime: float
    max_lifetime: float
    behavior: str
    properties: Dict


class AdvancedParticleSystem:
    """Разширена система за частици с комплексно поведение"""

    def __init__(self):
        self.config = PARTICLE_CONFIG
        self.logger = Logger()

        # Колекции от частици и емитери
        self.particles: List[Particle] = []
        self.emitters: List[Dict] = []

        # Физични параметри
        self.gravity = np.array([0.0, 98.1])
        self.wind = np.array([0.0, 0.0])

        # Поведенчески шаблони
        self.behaviors = {
            'standard': self._update_standard,
            'quantum': self._update_quantum,
            'vortex': self._update_vortex,
            'spiral': self._update_spiral,
            'attractor': self._update_attractor
        }

        # Метрики
        self.particles_created = 0
        self.particles_destroyed = 0

    def create_emitter(self, position: Tuple[float, float],
                       params: Dict) -> int:
        """Създава нов емитер на частици"""
        emitter = {
            'position': np.array(position),
            'params': params,
            'active': True,
            'last_emission': time.time(),
            'particles_emitted': 0
        }

        self.emitters.append(emitter)
        return len(self.emitters) - 1

    def update(self, delta_time: float):
        """Обновява системата от частици"""
        # Обновяване на емитерите
        self._update_emitters(delta_time)

        # Обновяване на частиците
        self._update_particles(delta_time)

        # Обработка на колизии
        self._handle_collisions()

        # Премахване на мъртви частици
        self._cleanup_particles()

    def _update_emitters(self, delta_time: float):
        """Обновява всички емитери"""
        current_time = time.time()

        for emitter in self.emitters:
            if not emitter['active']:
                continue

            # Проверка за време на емисия
            if (current_time - emitter['last_emission'] >
                    1.0 / emitter['params'].get('emission_rate', 10)):

                # Емитиране на нови частици
                num_particles = emitter['params'].get('burst_size', 1)
                for _ in range(num_particles):
                    self._emit_particle(emitter)

                emitter['last_emission'] = current_time

    def _emit_particle(self, emitter: Dict):
        """Емитира нова частица от емитер"""
        params = emitter['params']

        # Базови параметри
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(
            params.get('min_speed', 50),
            params.get('max_speed', 200)
        )

        # Създаване на частицата
        particle = Particle(
            position=emitter['position'].copy(),
            velocity=np.array([
                math.cos(angle) * speed,
                math.sin(angle) * speed
            ]),
            acceleration=np.zeros(2),
            color=self._generate_color(params),
            size=random.uniform(
                params.get('min_size', 2),
                params.get('max_size', 8)
            ),
            lifetime=random.uniform(
                params.get('min_lifetime', 1),
                params.get('max_lifetime', 3)
            ),
            max_lifetime=params.get('max_lifetime', 3),
            behavior=params.get('behavior', 'standard'),
            properties=params.get('properties', {})
        )

        self.particles.append(particle)
        self.particles_created += 1

    def _generate_color(self, params: Dict) -> Tuple[int, int, int, int]:
        """Генерира цвят за частица"""
        if 'color_gradient' in params:
            # Генериране на цвят от градиент
            gradient = params['color_gradient']
            t = random.random()
            start_color = np.array(gradient[0])
            end_color = np.array(gradient[1])
            color = start_color + (end_color - start_color) * t
            return tuple(map(int, color)) + (255,)
        else:
            # Генериране на случаен цвят в HSV пространството
            hue = random.random()
            saturation = random.uniform(0.8, 1.0)
            value = random.uniform(0.8, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            return tuple(int(x * 255) for x in rgb) + (255,)

    def _update_particles(self, delta_time: float):
        """Обновява всички частици"""
        for particle in self.particles:
            # Обновяване на времето на живот
            particle.lifetime -= delta_time

            # Прилагане на специфично поведение
            if particle.behavior in self.behaviors:
                self.behaviors[particle.behavior](particle, delta_time)

            # Обновяване на позицията
            particle.velocity += particle.acceleration * delta_time
            particle.position += particle.velocity * delta_time

            # Прилагане на глобални сили
            particle.velocity += (self.gravity + self.wind) * delta_time

            # Затихване
            particle.velocity *= 0.99

    def _update_quantum(self, particle: Particle, delta_time: float):
        """Квантово-подобно поведение"""
        # Вълнова функция
        phase = time.time() * particle.properties.get('frequency', 2.0)
        amplitude = particle.properties.get('amplitude', 10.0)

        particle.position += amplitude * np.array([
            math.sin(phase + particle.position[1] * 0.1),
            math.cos(phase + particle.position[0] * 0.1)
        ]) * delta_time

    def _update_vortex(self, particle: Particle, delta_time: float):
        """Вихрово движение"""
        center = particle.properties.get('center', np.array([400, 300]))
        to_center = center - particle.position
        distance = np.linalg.norm(to_center)

        if distance > 0:
            # Тангенциална сила
            tangent = np.array([-to_center[1], to_center[0]]) / distance
            strength = particle.properties.get('vortex_strength', 200.0)
            particle.acceleration = tangent * strength / (distance + 1.0)

    def render(self, surface: pygame.Surface):
        """Рендерира всички частици"""
        for particle in self.particles:
            # Изчисляване на алфа канала базирано на останалия живот
            alpha = int(255 * (particle.lifetime / particle.max_lifetime))
            color = particle.color[:3] + (alpha,)

            # Рендериране на частицата
            pos = tuple(map(int, particle.position))
            pygame.draw.circle(
                surface,
                color,
                pos,
                particle.size
            )

            # Добавяне на свечение ако е нужно
            if particle.properties.get('glow', False):
                self._render_glow(surface, particle)

    def _render_glow(self, surface: pygame.Surface, particle: Particle):
        """Рендерира свечение около частицата"""
        glow_size = particle.size * 2
        glow_surface = pygame.Surface((glow_size * 2, glow_size * 2),
                                      pygame.SRCALPHA)

        # Създаване на градиентно свечение
        for radius in range(int(glow_size), 0, -1):
            alpha = int(100 * (radius / glow_size))
            color = particle.color[:3] + (alpha,)
            pygame.draw.circle(
                glow_surface,
                color,
                (glow_size, glow_size),
                radius
            )

        # Прилагане на свечението
        pos = (
            int(particle.position[0] - glow_size),
            int(particle.position[1] - glow_size)
        )
        surface.blit(glow_surface, pos, special_flags=pygame.BLEND_ADD)
