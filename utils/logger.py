import logging
from typing import Dict, Any, Optional
import time
import json
from pathlib import Path
import traceback
import sys
from datetime import datetime
import threading
from queue import Queue
import numpy as np


class Logger:
    """Разширена система за логване с анализ и визуализация"""

    def __init__(self):
        self._setup_logging()

        # Опашка за асинхронно логване
        self.log_queue = Queue()
        self.is_running = True

        # Статистики
        self.stats = {
            'errors': 0,
            'warnings': 0,
            'performance_metrics': [],
            'events': [],
            'system_stats': {}
        }

        # Метрики за производителност
        self.performance_buffer = []
        self.performance_window = 100

        # Започване на логващата нишка
        self.log_thread = threading.Thread(target=self._process_log_queue)
        self.log_thread.daemon = True
        self.log_thread.start()

        # Създаване на директории за логове
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)

        # Файл за текущата сесия
        self.session_file = self.log_dir / \
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def _setup_logging(self):
        """Настройва системата за логване"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('game.log', mode='a')
            ]
        )
        self.logger = logging.getLogger('GameLogger')

    def _process_log_queue(self):
        """Обработва опашката за логване асинхронно"""
        while self.is_running:
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self.log_queue.task_done()
            except:
                continue

    def _write_log_entry(self, entry: Dict):
        """Записва лог запис във файл"""
        try:
            with open(self.session_file, 'a') as f:
                json.dump(entry, f)
                f.write('\n')
        except Exception as e:
            self.error(f"Failed to write log entry: {e}")

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Логва игрово събитие"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }

        self.log_queue.put(event)
        self.stats['events'].append(event)

    def log_performance(self, metrics: Dict[str, float]):
        """Логва метрики за производителност"""
        self.performance_buffer.append(metrics)

        if len(self.performance_buffer) > self.performance_window:
            self._analyze_performance()

    def _analyze_performance(self):
        """Анализира метрики за производителност"""
        if not self.performance_buffer:
            return

        # Изчисляване на статистики
        metrics_array = np.array([list(m.values())
                                 for m in self.performance_buffer])
        mean_metrics = np.mean(metrics_array, axis=0)
        std_metrics = np.std(metrics_array, axis=0)

        analysis = {
            'timestamp': time.time(),
            'mean_metrics': dict(zip(self.performance_buffer[0].keys(), mean_metrics)),
            'std_metrics': dict(zip(self.performance_buffer[0].keys(), std_metrics)),
            'samples': len(self.performance_buffer)
        }

        self.stats['performance_metrics'].append(analysis)
        self.performance_buffer = []

        # Проверка за проблеми с производителността
        self._check_performance_issues(analysis)

    def _check_performance_issues(self, analysis: Dict):
        """Проверява за проблеми с производителността"""
        thresholds = {
            'fps': 50,
            'frame_time': 20,  # ms
            'memory_usage': 0.8  # 80% използване
        }

        for metric, threshold in thresholds.items():
            if metric in analysis['mean_metrics']:
                value = analysis['mean_metrics'][metric]
                if ((metric == 'fps' and value < threshold) or
                        (metric != 'fps' and value > threshold)):
                    self.warning(f"Performance issue detected: {
                                 metric} = {value}")

    def info(self, message: str, extra: Optional[Dict] = None):
        """Логва информационно съобщение"""
        self.logger.info(message)
        self.log_event('info', {
            'message': message,
            'extra': extra or {}
        })

    def warning(self, message: str, extra: Optional[Dict] = None):
        """Логва предупреждение"""
        self.logger.warning(message)
        self.stats['warnings'] += 1
        self.log_event('warning', {
            'message': message,
            'extra': extra or {},
            'stack_trace': traceback.format_stack()
        })

    def error(self, message: str, extra: Optional[Dict] = None):
        """Логва грешка"""
        self.logger.error(message)
        self.stats['errors'] += 1
        self.log_event('error', {
            'message': message,
            'extra': extra or {},
            'stack_trace': traceback.format_exc()
        })

    def log_system_stats(self, stats: Dict):
        """Логва системни статистики"""
        self.stats['system_stats'] = {
            **self.stats['system_stats'],
            **stats,
            'timestamp': time.time()
        }

    def get_statistics(self) -> Dict:
        """Връща обобщени статистики"""
        return {
            'total_events': len(self.stats['events']),
            'error_rate': self.stats['errors'] / max(1, len(self.stats['events'])),
            'warning_rate': self.stats['warnings'] / max(1, len(self.stats['events'])),
            'performance_summary': self._get_performance_summary(),
            'system_stats': self.stats['system_stats']
        }

    def _get_performance_summary(self) -> Dict:
        """Генерира обобщение на производителността"""
        if not self.stats['performance_metrics']:
            return {}

        latest_metrics = self.stats['performance_metrics'][-10:]
        return {
            'recent_fps': np.mean([m['mean_metrics'].get('fps', 0)
                                   for m in latest_metrics]),
            'frame_time_stability': np.mean([m['std_metrics'].get('frame_time', 0)
                                             for m in latest_metrics]),
            'memory_trend': self._calculate_trend([m['mean_metrics'].get('memory_usage', 0)
                                                   for m in latest_metrics])
        }

    def _calculate_trend(self, values: list[float]) -> float:
        """Изчислява тренд в данните"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]

    def cleanup(self):
        """Почиства ресурсите при затваряне"""
        self.is_running = False
        self.log_thread.join()

        # Финален анализ
        if self.performance_buffer:
            self._analyze_performance()

        # Записване на финална статистика
        final_stats = {
            'timestamp': time.time(),
            'total_duration': time.time() - self.stats['events'][0]['timestamp']
            if self.stats['events'] else 0,
            'statistics': self.get_statistics()
        }

        with open(self.session_file, 'a') as f:
            json.dump(final_stats, f, indent=2)
