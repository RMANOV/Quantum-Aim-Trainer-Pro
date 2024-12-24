import pymunk
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math

from utils.logger import Logger
from config import PHYSICS_CONFIG


@dataclass
class PhysicsObject:
    """Физичен обект в симулацията"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    mass: float
    radius: float
    elasticity: float
    friction: float
    moment: float
    body: Optional[pymunk.Body] = None
    shape: Optional[pymunk.Shape] = None

    def create_body(self, space: pymunk.Space):
        """Създава физично тяло в пространството"""
        self.body = pymunk.Body(self.mass, self.moment)
        self.body.position = self.position
        self.body.velocity = self.velocity

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = self.elasticity
        self.shape.friction = self.friction

        space.add(self.body, self.shape)


class PhysicsEngine:
    """Физична симулация с напреднали възможности"""

    def __init__(self):
        self.config = PHYSICS_CONFIG
        self.logger = Logger()

        # Инициализация на физичното пространство
        self.space = pymunk.Space()
        self.space.gravity = self.config.gravity
        self.space.damping = self.config.damping

        # Колекции от обекти
        self.objects: List[PhysicsObject] = []
        self.force_fields: List[Dict] = []
        self.constraints: List[pymunk.Constraint] = []

        # Collision handlers
        self._setup_collision_handlers()

        # Performance tracking
        self.physics_time = 0.0
        self.collision_count = 0

    def _setup_collision_handlers(self):
        """Настройва обработката на колизии"""
        # Основен handler за колизии
        handler = self.space.add_default_collision_handler()
        handler.begin = self._on_collision_begin
        handler.separate = self._on_collision_separate

        # Специфични handlers за различни типове обекти
        for type_a in self.config.collision_types:
            for type_b in self.config.collision_types:
                if type_a != type_b:
                    self._add_specific_handler(type_a, type_b)

    def _add_specific_handler(self, type_a: int, type_b: int):
        """Добавя специфичен handler за двойка типове обекти"""
        handler = self.space.add_collision_handler(type_a, type_b)
        handler.begin = lambda arbiter, space, data: self._handle_specific_collision(
            arbiter, type_a, type_b
        )

    def add_object(self, obj: PhysicsObject) -> int:
        """Добавя нов обект във физичната симулация"""
        obj.create_body(self.space)
        self.objects.append(obj)
        return len(self.objects) - 1

    def add_force_field(self, position: Tuple[float, float],
                        radius: float, strength: float,
                        field_type: str = 'radial'):
        """Добавя силово поле"""
        force_field = {
            'position': position,
            'radius': radius,
            'strength': strength,
            'type': field_type,
            'active': True
        }
        self.force_fields.append(force_field)

    def update(self, delta_time: float):
        """Обновява физичната симулация"""
        # Разделяме delta_time на по-малки стъпки за по-точна симулация
        steps = max(1, int(delta_time / self.config.min_step))
        step_dt = delta_time / steps

        for _ in range(steps):
            # Прилагане на силови полета
            self._apply_force_fields()

            # Прилагане на въздушно съпротивление
            self._apply_air_resistance()

            # Обновяване на физиката
            self.space.step(step_dt)

        # Обновяване на метрики
        self._update_metrics()

    def _apply_force_fields(self):
        """Прилага силовите полета върху обектите"""
        for field in self.force_fields:
            if not field['active']:
                continue

            for obj in self.objects:
                if not obj.body:
                    continue

                # Изчисляване на разстоянието до полето
                dx = obj.body.position.x - field['position'][0]
                dy = obj.body.position.y - field['position'][1]
                distance = math.sqrt(dx*dx + dy*dy)

                if distance < field['radius']:
                    # Изчисляване на силата
                    force_magnitude = field['strength'] * \
                        (1 - distance/field['radius'])

                    if field['type'] == 'radial':
                        # Радиална сила (привличане/отблъскване)
                        force = (dx/distance * force_magnitude,
                                 dy/distance * force_magnitude)
                    elif field['type'] == 'vortex':
                        # Въртяща сила
                        force = (-dy/distance * force_magnitude,
                                 dx/distance * force_magnitude)

                    obj.body.apply_force_at_local_point(force)

    def _apply_air_resistance(self):
        """Прилага въздушно съпротивление към обектите"""
        for obj in self.objects:
            if not obj.body:
                continue

            velocity = obj.body.velocity
            speed_squared = velocity.x**2 + velocity.y**2

            if speed_squared > 0:
                resistance = -self.config.air_resistance * speed_squared
                direction = velocity.normalized()
                force = (direction.x * resistance, direction.y * resistance)
                obj.body.apply_force_at_local_point(force)

    def _on_collision_begin(self, arbiter, space, data):
        """Обработка на началото на колизия"""
        self.collision_count += 1
        return True

    def _handle_specific_collision(self, arbiter, type_a: int, type_b: int):
        """Обработка на специфични колизии между типове обекти"""
        # Специфична логика за различни комбинации от типове
        return True

    def _update_metrics(self):
        """Обновява метриките на физичната симулация"""
        # Тук можем да добавим повече метрики при нужда
        pass

    def get_object_state(self, index: int) -> Optional[Dict]:
        """Връща текущото състояние на обект"""
        if 0 <= index < len(self.objects):
            obj = self.objects[index]
            if obj.body:
                return {
                    'position': obj.body.position,
                    'velocity': obj.body.velocity,
                    'angle': obj.body.angle,
                    'angular_velocity': obj.body.angular_velocity
                }
        return None

    def clear(self):
        """Изчиства всички обекти от симулацията"""
        for obj in self.objects:
            if obj.body:
                self.space.remove(obj.body, obj.shape)
        self.objects.clear()
        self.force_fields.clear()
        self.constraints.clear()
