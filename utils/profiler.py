import time
from typing import Dict, List, Optional
import cProfile
import pstats
from functools import wraps
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
import threading
import psutil
import gputil


class Profiler:
    """Разширен профайлер за анализ на производителността"""

    def __init__(self):
        self.profiler = cProfile.Profile()
        self.timings = defaultdict(list)
        self.memory_usage = []
        self.gpu_usage = []
        self.frame_times = []
        self.active_sections = {}
        self.thread_timings = defaultdict(lambda: defaultdict(list))

        # Флаг за активно профилиране
        self.is_profiling = False

        # Статистики за критични секции
        self.critical_sections = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'samples': []
        })

        # Започване на мониторинг нишка
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def start(self):
        """Стартира профилирането"""
        self.is_profiling = True
        self.profiler.enable()
        self.start_time = time.time()

    def stop(self):
        """Спира профилирането"""
        self.is_profiling = False
        self.profiler.disable()
        self.end_time = time.time()

    @contextmanager
    def measure(self, section_name: str):
        """Контекст мениджър за измерване на секции от кода"""
        start_time = time.perf_counter()
        thread_id = threading.get_ident()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Запазване на времето за секцията
            self.timings[section_name].append(duration)
            self.thread_timings[thread_id][section_name].append(duration)

            # Обновяване на статистиките за критичната секция
            stats = self.critical_sections[section_name]
            stats['calls'] += 1
            stats['total_time'] += duration
            stats['max_time'] = max(stats['max_time'], duration)
            stats['min_time'] = min(stats['min_time'], duration)
            stats['samples'].append(duration)

            # Запазване само на последните 1000 измервания
            if len(stats['samples']) > 1000:
                stats['samples'] = stats['samples'][-1000:]

    def _monitor_system_resources(self):
        """Мониторинг на системните ресурси"""
        while True:
            if self.is_profiling:
                # CPU и памет
                process = psutil.Process()
                self.memory_usage.append(process.memory_percent())

                # GPU използване
                try:
                    gpus = gputil.getGPUs()
                    if gpus:
                        gpu_load = np.mean([gpu.load for gpu in gpus])
                        self.gpu_usage.append(gpu_load)
                except:
                    pass

            time.sleep(1)

    def add_frame_time(self, frame_time: float):
        """Добавя време за кадър"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 1000:
            self.frame_times = self.frame_times[-1000:]

    def get_statistics(self) -> Dict:
        """Връща подробни статистики"""
        stats = {}

        # Общи статистики
        stats['total_time'] = self.end_time - self.start_time
        stats['sections'] = {}

        # Статистики за секциите
        for section, timings in self.timings.items():
            if timings:
                stats['sections'][section] = {
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'calls': len(timings),
                    'total_time': np.sum(timings),
                    'percentage': (np.sum(timings) / stats['total_time']) * 100
                }

        # Статистики за кадрите
        if self.frame_times:
            stats['frame_stats'] = {
                'mean_fps': 1.0 / np.mean(self.frame_times),
                'std_fps': np.std([1.0 / t for t in self.frame_times]),
                'frame_time_mean': np.mean(self.frame_times) * 1000,  # в ms
                'frame_time_std': np.std(self.frame_times) * 1000,    # в ms
                '99th_percentile': np.percentile(self.frame_times, 99) * 1000
            }

        # Системни ресурси
        if self.memory_usage:
            stats['system'] = {
                'mean_memory': np.mean(self.memory_usage),
                'max_memory': np.max(self.memory_usage),
                'mean_gpu': np.mean(self.gpu_usage) if self.gpu_usage else None,
                'max_gpu': np.max(self.gpu_usage) if self.gpu_usage else None
            }

        # Критични секции
        stats['critical_sections'] = {}
        for section, section_stats in self.critical_sections.items():
            if section_stats['calls'] > 0:
                samples = section_stats['samples']
                stats['critical_sections'][section] = {
                    'calls': section_stats['calls'],
                    'mean_time': section_stats['total_time'] / section_stats['calls'],
                    'max_time': section_stats['max_time'],
                    'min_time': section_stats['min_time'],
                    'std_time': np.std(samples) if len(samples) > 1 else 0,
                    'recent_trend': self._calculate_trend(samples[-100:]) if len(samples) >= 100 else 0
                }

        return stats

    def _calculate_trend(self, values: List[float]) -> float:
        """Изчислява тренд в данните"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]

    def generate_report(self, output_file: Optional[str] = None):
        """Генерира подробен отчет"""
        stats = self.get_statistics()

        report = ["Performance Profiling Report", "=" * 30, ""]

        # Общо време
        report.append(f"Total profiling time: {
                      stats['total_time']:.2f} seconds\n")

        # Секции
        report.append("Section Statistics:")
        report.append("-" * 20)
        for section, section_stats in stats['sections'].items():
            report.append(f"\n{section}:")
            report.append(f"  Calls: {section_stats['calls']}")
            report.append(f"  Mean time: {section_stats['mean']*1000:.2f} ms")
            report.append(f"  Std dev: {section_stats['std']*1000:.2f} ms")
            report.append(f"  Total time: {
                          section_stats['total_time']:.2f} seconds")
            report.append(f"  % of total: {section_stats['percentage']:.1f}%")

        # Статистики за кадрите
        if 'frame_stats' in stats:
            report.append("\nFrame Statistics:")
            report.append("-" * 20)
            report.append(f"Mean FPS: {stats['frame_stats']['mean_fps']:.1f}")
            report.append(f"Frame time (mean): {
                          stats['frame_stats']['frame_time_mean']:.2f} ms")
            report.append(f"Frame time (99th): {
                          stats['frame_stats']['99th_percentile']:.2f} ms")

        # Системни ресурси
        if 'system' in stats:
            report.append("\nSystem Resources:")
            report.append("-" * 20)
            report.append(f"Mean memory usage: {
                          stats['system']['mean_memory']:.1f}%")
            if stats['system']['mean_gpu'] is not None:
                report.append(f"Mean GPU usage: {
                              stats['system']['mean_gpu']:.1f}%")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

        return report_text

    def profile_function(self, func):
        """Декоратор за профилиране на функция"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.measure(func.__name__):
                return func(*args, **kwargs)
        return wrapper
