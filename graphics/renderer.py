import pygame
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import moderngl
import math
from PIL import Image
import colorsys

from utils.logger import Logger
from config import GRAPHICS_CONFIG


@dataclass
class RenderObject:
    """Обект за рендериране"""
    position: Tuple[float, float]
    size: float
    color: Tuple[int, int, int, int]
    glow: float
    rotation: float
    effects: List[str]
    z_index: int
    shader_params: Dict


class AdvancedRenderer:
    """Разширен рендерер с поддръжка на модерни графични ефекти"""

    def __init__(self, screen: pygame.Surface):
        self.config = GRAPHICS_CONFIG
        self.logger = Logger()
        self.screen = screen
        self.screen_size = screen.get_size()

        # Инициализация на ModernGL контекст
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self._setup_framebuffer()

        # Зареждане на шейдъри
        self.shaders = self._load_shaders()

        # Слоеве за рендериране
        self.render_layers: Dict[int, List[RenderObject]] = {}
        self.post_process_effects = []

        # Кеш за текстури
        self.texture_cache = {}

        # Метрики
        self.frame_time = 0.0
        self.draw_calls = 0

    def _setup_framebuffer(self) -> moderngl.Framebuffer:
        """Настройва framebuffer за рендериране"""
        color_texture = self.ctx.texture(self.screen_size, 4)
        depth_texture = self.ctx.depth_texture(self.screen_size)
        return self.ctx.framebuffer(
            color_attachments=[color_texture],
            depth_attachment=depth_texture
        )

    def _load_shaders(self) -> Dict[str, moderngl.Program]:
        """Зарежда и компилира шейдърите"""
        shaders = {}

        # Базов шейдър
        shaders['basic'] = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 uv;
                uniform vec2 scale;
                uniform vec2 position;
                uniform float rotation;
                
                void main() {
                    vec2 pos = in_position;
                    float c = cos(rotation);
                    float s = sin(rotation);
                    pos = vec2(
                        pos.x * c - pos.y * s,
                        pos.x * s + pos.y * c
                    );
                    pos = pos * scale + position;
                    gl_Position = vec4(pos, 0.0, 1.0);
                    uv = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 uv;
                out vec4 fragColor;
                uniform sampler2D texture0;
                uniform vec4 color;
                uniform float glow;
                
                void main() {
                    vec4 texColor = texture(texture0, uv);
                    vec4 finalColor = texColor * color;
                    finalColor.rgb += glow * vec3(1.0);
                    fragColor = finalColor;
                }
            '''
        )

        # Шейдър за bloom ефект
        shaders['bloom'] = self.ctx.program(
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
                in vec2 uv;
                out vec4 fragColor;
                uniform sampler2D texture0;
                uniform float intensity;
                uniform vec2 direction;
                
                void main() {
                    vec4 color = vec4(0.0);
                    float total_weight = 0.0;
                    
                    for(float i = -4.0; i <= 4.0; i++) {
                        float weight = exp(-0.5 * i * i);
                        vec2 offset = direction * i / textureSize(texture0, 0);
                        color += texture(texture0, uv + offset) * weight;
                        total_weight += weight;
                    }
                    
                    fragColor = color / total_weight * intensity;
                }
            '''
        )

        return shaders

    def add_object(self, obj: RenderObject):
        """Добавя обект за рендериране"""
        if obj.z_index not in self.render_layers:
            self.render_layers[obj.z_index] = []
        self.render_layers[obj.z_index].append(obj)

    def clear_objects(self):
        """Изчиства всички обекти"""
        self.render_layers.clear()

    def render(self):
        """Основен метод за рендериране"""
        self.draw_calls = 0

        # Изчистване на framebuffer-а
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Рендериране по слоеве
        for z_index in sorted(self.render_layers.keys()):
            self._render_layer(self.render_layers[z_index])

        # Пост-процесинг ефекти
        self._apply_post_processing()

        # Прехвърляне към pygame повърхността
        self._transfer_to_pygame()

    def _render_layer(self, objects: List[RenderObject]):
        """Рендерира всички обекти в един слой"""
        for obj in objects:
            self._render_object(obj)
            self.draw_calls += 1

    def _render_object(self, obj: RenderObject):
        """Рендерира единичен обект"""
        # Избор на шейдър
        shader = self.shaders['basic']
        shader['scale'].value = (obj.size / self.screen_size[0],
                                 obj.size / self.screen_size[1])
        shader['position'].value = (obj.position[0] / self.screen_size[0] * 2 - 1,
                                    obj.position[1] / self.screen_size[1] * 2 - 1)
        shader['rotation'].value = obj.rotation
        shader['color'].value = tuple(c/255 for c in obj.color)
        shader['glow'].value = obj.glow

        # Прилагане на специални ефекти
        for effect in obj.effects:
            self._apply_effect(obj, effect)

    def _apply_post_processing(self):
        """Прилага пост-процесинг ефекти"""
        for effect in self.post_process_effects:
            if effect == 'bloom':
                self._apply_bloom()

    def _apply_bloom(self):
        """Прилага bloom ефект"""
        bloom_shader = self.shaders['bloom']

        # Хоризонтален блур
        bloom_shader['direction'].value = (1.0/self.screen_size[0], 0.0)
        bloom_shader['intensity'].value = self.config.bloom_intensity

        # Вертикален блур
        bloom_shader['direction'].value = (0.0, 1.0/self.screen_size[1])

    def _transfer_to_pygame(self):
        """Прехвърля резултата към pygame повърхността"""
        # Четене на пикселите от framebuffer-а
        pixels = self.fbo.read(components=4)

        # Конвертиране към pygame повърхност
        surface = pygame.image.fromstring(
            pixels,
            self.screen_size,
            'RGBA'
        )

        # Изобразяване върху екрана
        self.screen.blit(surface, (0, 0))

    def add_post_process_effect(self, effect: str):
        """Добавя пост-процесинг ефект"""
        if effect not in self.post_process_effects:
            self.post_process_effects.append(effect)

    def get_metrics(self) -> Dict:
        """Връща метрики за рендерирането"""
        return {
            'frame_time': self.frame_time,
            'draw_calls': self.draw_calls,
            'active_layers': len(self.render_layers),
            'total_objects': sum(len(layer) for layer in self.render_layers.values())
        }
