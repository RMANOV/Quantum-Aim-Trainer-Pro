import pygame
import moderngl
from typing import Dict, Optional, Any
import threading
from pathlib import Path
import json
import numpy as np
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor
import logging


class ResourceManager:
    """Мениджър за ресурси с асинхронно зареждане и кеширане"""

    def __init__(self, resource_dir: str = "assets"):
        self.resource_dir = Path(resource_dir)

        # Речници за ресурси
        self.textures: Dict[str, pygame.Surface] = {}
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.shaders: Dict[str, Any] = {}
        self.fonts: Dict[str, Dict[int, pygame.font.Font]] = {}

        # ModernGL контекст за шейдъри
        self.ctx = moderngl.create_standalone_context()

        # Кеш система
        self.cache = ResourceCache()

        # Асинхронно зареждане
        self.loader = AsyncResourceLoader(self.resource_dir)

        # Проследяване на използването
        self.usage_stats = ResourceUsageTracker()

        # Инициализация
        self._initialize()

    def _initialize(self):
        """Инициализира основните ресурси"""
        # Зареждане на манифест файл
        manifest_path = self.resource_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)

            # Асинхронно зареждане на критичните ресурси
            self.loader.load_resources(
                self.manifest.get("critical", []),
                callback=self._on_critical_loaded
            )

    def _on_critical_loaded(self, resources: Dict[str, Any]):
        """Callback при зареждане на критичните ресурси"""
        for resource_id, resource in resources.items():
            self._store_resource(resource_id, resource)

    def _store_resource(self, resource_id: str, resource: Any):
        """Съхранява ресурс в подходящата колекция"""
        resource_type = resource_id.split('/')[0]

        if resource_type == 'textures':
            self.textures[resource_id] = resource
        elif resource_type == 'sounds':
            self.sounds[resource_id] = resource
        elif resource_type == 'shaders':
            self.shaders[resource_id] = resource
        elif resource_type == 'fonts':
            font_name = resource_id.split('/')[-1]
            self.fonts[font_name] = resource

        # Обновяване на кеша
        self.cache.add(resource_id, resource)

    def load_texture(self, texture_id: str) -> Optional[pygame.Surface]:
        """Зарежда текстура"""
        if texture_id in self.textures:
            self.usage_stats.record_usage(texture_id)
            return self.textures[texture_id]

        # Проверка в кеша
        if self.cache.has(texture_id):
            return self.cache.get(texture_id)

        # Асинхронно зареждане
        self.loader.load_resource(
            texture_id,
            callback=lambda r: self._store_resource(texture_id, r)
        )
        return None

    def load_shader(self, shader_id: str) -> Optional[moderngl.Program]:
        """Зарежда и компилира шейдър"""
        if shader_id in self.shaders:
            self.usage_stats.record_usage(shader_id)
            return self.shaders[shader_id]

        vertex_path = self.resource_dir / "shaders" / f"{shader_id}.vert"
        fragment_path = self.resource_dir / "shaders" / f"{shader_id}.frag"

        try:
            with open(vertex_path) as v, open(fragment_path) as f:
                program = self.ctx.program(
                    vertex_shader=v.read(),
                    fragment_shader=f.read()
                )
                self.shaders[shader_id] = program
                return program
        except Exception as e:
            logging.error(f"Failed to load shader {shader_id}: {e}")
            return None

    def get_font(self, font_name: str, size: int) -> pygame.font.Font:
        """Получава шрифт с определен размер"""
        if font_name in self.fonts and size in self.fonts[font_name]:
            self.usage_stats.record_usage(f"fonts/{font_name}/{size}")
            return self.fonts[font_name][size]

        try:
            font_path = self.resource_dir / "fonts" / f"{font_name}.ttf"
            font = pygame.font.Font(str(font_path), size)

            if font_name not in self.fonts:
                self.fonts[font_name] = {}
            self.fonts[font_name][size] = font

            return font
        except Exception as e:
            logging.error(f"Failed to load font {font_name}: {e}")
            return pygame.font.SysFont('arial', size)

    def preload_resources(self, resource_ids: List[str]):
        """Предварително зареждане на ресурси"""
        self.loader.load_resources(resource_ids)

    def cleanup(self):
        """Освобождава ресурсите"""
        # Освобождаване на текстури
        for texture in self.textures.values():
            if hasattr(texture, 'free'):
                texture.free()

        # Освобождаване на звуци
        for sound in self.sounds.values():
            sound.stop()

        # Освобождаване на шейдъри
        for shader in self.shaders.values():
            shader.release()

        # Освобождаване на кеша
        self.cache.clear()


class ResourceCache:
    """Кеш система за ресурси"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}

    def add(self, key: str, value: Any):
        """Добавя ресурс в кеша"""
        if len(self.cache) >= self.max_size:
            self._cleanup()

        self.cache[key] = value
        self.access_times[key] = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Получава ресурс от кеша"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def has(self, key: str) -> bool:
        """Проверява за наличие на ресурс"""
        return key in self.cache

    def _cleanup(self):
        """Почиства най-старите ресурси"""
        if not self.access_times:
            return

        oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest]
        del self.access_times[oldest]

    def clear(self):
        """Изчиства кеша"""
        self.cache.clear()
        self.access_times.clear()


class AsyncResourceLoader:
    """Асинхронен зареждач на ресурси"""

    def __init__(self, resource_dir: Path):
        self.resource_dir = resource_dir
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loading_futures = {}

    def load_resource(self, resource_id: str, callback=None):
        """Зарежда единичен ресурс асинхронно"""
        if resource_id in self.loading_futures:
            return

        future = self.executor.submit(self._load_resource, resource_id)
        if callback:
            future.add_done_callback(
                lambda f: callback(f.result())
            )
        self.loading_futures[resource_id] = future

    def load_resources(self, resource_ids: List[str], callback=None):
        """Зарежда множество ресурси асинхронно"""
        futures = []
        for resource_id in resource_ids:
            if resource_id not in self.loading_futures:
                future = self.executor.submit(self._load_resource, resource_id)
                self.loading_futures[resource_id] = future
                futures.append(future)

        if callback:
            def on_all_completed(results):
                resources = {
                    resource_id: result
                    for resource_id, result in zip(resource_ids, results)
                }
                callback(resources)

            self._wait_for_futures(futures, on_all_completed)

    def _load_resource(self, resource_id: str) -> Any:
        """Зарежда ресурс според типа му"""
        try:
            resource_type = resource_id.split('/')[0]

            if resource_type == 'textures':
                return self._load_texture(resource_id)
            elif resource_type == 'sounds':
                return self._load_sound(resource_id)
            elif resource_type == 'fonts':
                return self._load_font(resource_id)

        except Exception as e:
            logging.error(f"Failed to load resource {resource_id}: {e}")
            return None

    def _load_texture(self, texture_id: str) -> pygame.Surface:
        """Зарежда текстура"""
        path = self.resource_dir / "textures" / f"{texture_id}.png"
        return pygame.image.load(str(path)).convert_alpha()

    def _load_sound(self, sound_id: str) -> pygame.mixer.Sound:
        """Зарежда звук"""
        path = self.resource_dir / "sounds" / f"{sound_id}.wav"
        return pygame.mixer.Sound(str(path))

    def _load_font(self, font_id: str) -> Dict[int, pygame.font.Font]:
        """Зарежда шрифт"""
        path = self.resource_dir / "fonts" / f"{font_id}.ttf"
        return {
            size: pygame.font.Font(str(path), size)
            for size in [8, 12, 16, 24, 32, 48, 64]
        }

    @staticmethod
    def _wait_for_futures(futures, callback):
        """Изчаква завършването на множество futures"""
        def check_futures():
            if all(f.done() for f in futures):
                results = [f.result() for f in futures]
                callback(results)
                return
            threading.Timer(0.1, check_futures).start()

        check_futures()


class ResourceUsageTracker:
    """Проследява използването на ресурсите"""

    def __init__(self):
        self.usage_counts: Dict[str, int] = {}
        self.last_used: Dict[str, float] = {}

    def record_usage(self, resource_id: str):
        """Записва използване на ресурс"""
        self.usage_counts[resource_id] = self.usage_counts.get(
            resource_id, 0) + 1
        self.last_used[resource_id] = time.time()

    def get_stats(self) -> Dict:
        """Връща статистика за използването"""
        return {
            'total_resources': len(self.usage_counts),
            'most_used': sorted(
                self.usage_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'least_used': sorted(
                self.usage_counts.items(),
                key=lambda x: x[1]
            )[:10],
            'unused_time': {
                resource_id: time.time() - last_used
                for resource_id, last_used in self.last_used.items()
            }
        }
