import pygame
import numpy as np
from typing import List, Dict, Tuple, Optional
import moderngl
import math
from scipy import signal
from dataclasses import dataclass
import time

from utils.logger import Logger
from config import EFFECTS_CONFIG


@dataclass
class VisualEffect:
    """Структура за визуален ефект"""
    effect_type: str
    intensity: float
    duration: float
    start_time: float
    parameters: Dict
    affected_area: Optional[Tuple[int, int, int, int]] = None


class VisualEffectsProcessor:
    """Процесор за сложни визуални ефекти"""

    def __init__(self, screen_size: Tuple[int, int]):
        self.config = EFFECTS_CONFIG
        self.logger = Logger()
        self.screen_size = screen_size

        # ModernGL контекст
        self.ctx = moderngl.create_standalone_context()

        # Шейдъри за ефекти
        self.shaders = self._initialize_shaders()

        # Активни ефекти
        self.active_effects: List[VisualEffect] = []

        # Буфери за пост-процесинг
        self.buffers = self._create_buffers()

        # Параметри за времето
        self.current_time = 0.0
        self.last_update = time.time()

    def _initialize_shaders(self) -> Dict[str, moderngl.Program]:
        """Инициализира шейдърите за различните ефекти"""
        shaders = {}

        # Хроматична аберация
        shaders['chromatic'] = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 uv;
                
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    uv = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                uniform float intensity;
                in vec2 uv;
                out vec4 fragColor;
                
                void main() {
                    vec2 offset = (uv - 0.5) * intensity;
                    vec4 r = texture(texture0, uv + offset);
                    vec4 g = texture(texture0, uv);
                    vec4 b = texture(texture0, uv - offset);
                    fragColor = vec4(r.r, g.g, b.b, 1.0);
                }
            '''
        )

        # Волнов ефект
        shaders['wave'] = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 uv;
                
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    uv = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                uniform float time;
                uniform float amplitude;
                uniform float frequency;
                in vec2 uv;
                out vec4 fragColor;
                
                void main() {
                    vec2 offset = vec2(
                        amplitude * sin(uv.y * frequency + time),
                        amplitude * cos(uv.x * frequency + time)
                    );
                    fragColor = texture(texture0, uv + offset);
                }
            '''
        )

        # Глоу ефект
        shaders['glow'] = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 uv;
                
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    uv = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                uniform float intensity;
                uniform vec3 glow_color;
                in vec2 uv;
                out vec4 fragColor;
                
                void main() {
                    vec4 color = texture(texture0, uv);
                    float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                    vec3 glow = glow_color * brightness * intensity;
                    fragColor = vec4(color.rgb + glow, color.a);
                }
            '''
        )

        return shaders

    def _create_buffers(self) -> Dict:
        """Създава буфери за пост-процесинг"""
        buffers = {}

        # Основен буфер
        buffers['main'] = self.ctx.texture(self.screen_size, 4)

        # Буфер за размиване
        buffers['blur'] = self.ctx.texture(self.screen_size, 4)

        # Буфер за ефекти
        buffers['effects'] = self.ctx.texture(self.screen_size, 4)

        return buffers

    def add_effect(self, effect_type: str, intensity: float,
                   duration: float, parameters: Dict = None):
        """Добавя нов визуален ефект"""
        effect = VisualEffect(
            effect_type=effect_type,
            intensity=intensity,
            duration=duration,
            start_time=self.current_time,
            parameters=parameters or {}
        )

        self.active_effects.append(effect)

    def process_frame(self, surface: pygame.Surface) -> pygame.Surface:
        """Обработва кадър с всички активни ефекти"""
        # Обновяване на времето
        self.current_time = time.time()
        delta_time = self.current_time - self.last_update

        # Конвертиране на pygame повърхността към текстура
        texture_data = pygame.image.tostring(surface, 'RGBA')
        self.buffers['main'].write(texture_data)

        # Прилагане на ефектите
        result = self._apply_effects(delta_time)

        # Премахване на изтекли ефекти
        self._cleanup_effects()

        self.last_update = self.current_time
        return result

    def _apply_effects(self, delta_time: float) -> pygame.Surface:
        """Прилага всички активни ефекти"""
        result = self.buffers['main']

        for effect in self.active_effects:
            # Изчисляване на интензивността базирана на времето
            progress = (self.current_time - effect.start_time) / \
                effect.duration
            current_intensity = effect.intensity * (1.0 - progress)

            if effect.effect_type == 'chromatic':
                result = self._apply_chromatic_aberration(
                    result,
                    current_intensity
                )
            elif effect.effect_type == 'wave':
                result = self._apply_wave_effect(
                    result,
                    current_intensity,
                    effect.parameters.get('frequency', 10.0)
                )
            elif effect.effect_type == 'glow':
                result = self._apply_glow_effect(
                    result,
                    current_intensity,
                    effect.parameters.get('color', (1.0, 1.0, 1.0))
                )

        # Конвертиране обратно към pygame повърхност
        return self._texture_to_surface(result)

    def _cleanup_effects(self):
        """Премахва изтеклите ефекти"""
        self.active_effects = [
            effect for effect in self.active_effects
            if self.current_time - effect.start_time < effect.duration
        ]

    def _texture_to_surface(self, texture) -> pygame.Surface:
        """Конвертира OpenGL текстура към pygame повърхност"""
        data = texture.read()
        return pygame.image.fromstring(
            data,
            self.screen_size,
            'RGBA'
        )
