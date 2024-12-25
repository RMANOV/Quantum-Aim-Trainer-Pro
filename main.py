import pygame
import sys
from typing import Optional
import time
import traceback
import argparse

from core.game_engine import GameEngine
from utils.logger import Logger
from utils.profiler import Profiler
from utils.config_loader import ConfigManager
from utils.resource_manager import ResourceManager


class GameLauncher:
    """Основен клас за стартиране и управление на играта"""

    def __init__(self):
        self.logger = Logger()
        self.profiler = Profiler()

        try:
            # Инициализация на основните системи
            self._initialize_systems()

            # Зареждане на конфигурации и ресурси
            self._load_configurations()
            self._initialize_resources()

            # Създаване на game engine
            self.engine = self._create_game_engine()

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _initialize_systems(self):
        """Инициализира основните системи"""
        self.logger.info("Initializing core systems...")

        # Инициализация на pygame
        pygame.init()
        pygame.mixer.init()

        # Проверка на системните възможности
        self._check_system_capabilities()

    def _check_system_capabilities(self):
        """Проверява системните възможности"""
        # Проверка на дисплея
        display_info = pygame.display.Info()
        self.logger.info(f"Display: {display_info.current_w}x{
                         display_info.current_h}")

        # Проверка на OpenGL поддръжка
        try:
            import moderngl
            self.logger.info("OpenGL support: Available")
        except ImportError:
            self.logger.warning("OpenGL support: Not available")

        # Проверка на аудио системата
        if not pygame.mixer.get_init():
            self.logger.warning("Audio system initialization failed")

    def _load_configurations(self):
        """Зарежда конфигурациите"""
        self.logger.info("Loading configurations...")
        self.config_manager = ConfigManager()

        # Парсване на аргументи от командния ред
        parser = argparse.ArgumentParser(description='Quantum Aim Trainer')
        parser.add_argument('--profile', help='Configuration profile to use')
        parser.add_argument('--fullscreen', action='store_true',
                            help='Run in fullscreen mode')
        args = parser.parse_args()

        # Прилагане на профил ако е указан
        if args.profile:
            if not self.config_manager.load_profile(args.profile):
                self.logger.warning(
                    f"Profile {args.profile} not found, using defaults")

        # Прилагане на командни аргументи
        if args.fullscreen:
            self.config_manager.graphics.fullscreen = True

    def _initialize_resources(self):
        """Инициализира resource manager и зарежда ресурсите"""
        self.logger.info("Initializing resource manager...")
        self.resource_manager = ResourceManager()

        # Предварително зареждане на критичните ресурси
        critical_resources = [
            "textures/target",
            "textures/particle",
            "sounds/shoot",
            "sounds/hit",
            "fonts/main"
        ]
        self.resource_manager.preload_resources(critical_resources)

    def _create_game_engine(self) -> GameEngine:
        """Създава и конфигурира game engine"""
        self.logger.info("Creating game engine...")

        return GameEngine(
            config=self.config_manager,
            resource_manager=self.resource_manager,
            logger=self.logger,
            profiler=self.profiler
        )

    def run(self):
        """Стартира главния цикъл на играта"""
        self.logger.info("Starting game...")
        self.profiler.start()

        try:
            # Главен цикъл
            self.engine.run()

        except Exception as e:
            self.logger.error(f"Game crashed: {e}")
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Почиства ресурсите и затваря играта"""
        self.logger.info("Cleaning up...")

        # Спиране на профилирането и генериране на отчет
        self.profiler.stop()
        report = self.profiler.generate_report("profile_report.txt")
        self.logger.info("Profile report generated")

        # Записване на конфигурациите
        self.config_manager.save_config()

        # Освобождаване на ресурсите
        self.resource_manager.cleanup()

        # Освобождаване на логъра
        self.logger.cleanup()

        # Изключване на pygame
        pygame.quit()

    @staticmethod
    def display_error_window(error_message: str):
        """Показва прозорец с грешка"""
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", error_message)
        root.destroy()


def main():
    """Входна точка на програмата"""
    try:
        # Създаване и стартиране на играта
        game = GameLauncher()
        game.run()

    except Exception as e:
        # Показване на грешката на потребителя
        error_message = f"Fatal error: {str(e)}\n\nCheck logs for details."
        GameLauncher.display_error_window(error_message)

        # Запис на грешката в лог файл
        with open("crash_log.txt", "w") as f:
            f.write(f"Crash time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(traceback.format_exc())

        sys.exit(1)

    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()


# python main.py --profile performance --fullscreen
# python main.py --profile performance
# python main.py --fullscreen
# python main.py --profile default
# python main.py --profile default --fullscreen