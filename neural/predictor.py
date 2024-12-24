import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple, Optional, Dict
import time
from collections import deque

from utils.logger import Logger
from config import NEURAL_CONFIG


class NeuralPredictionNetwork:
    """Невронна мрежа за предсказване на поведението на играча"""

    def __init__(self):
        self.config = NEURAL_CONFIG
        self.logger = Logger()

        # Параметри на мрежата
        self.sequence_length = self.config.sequence_length
        self.feature_dimension = self.config.feature_dimension
        self.hidden_units = self.config.hidden_units

        # Буфери за данни
        self.state_buffer = deque(maxlen=self.sequence_length * 2)
        self.prediction_history = deque(maxlen=1000)

        # Метрики
        self.prediction_accuracy = 0.0
        self.confidence_scores = deque(maxlen=100)
        self.training_losses = []

        # Създаване на модела
        self.model = self._build_model()
        self.target_model = self._build_model()  # За стабилно обучение

        # Timestamp на последното обновяване
        self.last_update_time = time.time()

    def _build_model(self) -> Sequential:
        """Създава архитектурата на невронната мрежа"""
        model = Sequential([
            LSTM(self.hidden_units[0],
                 input_shape=(self.sequence_length, self.feature_dimension),
                 return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(self.hidden_units[1], return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(self.hidden_units[2]),
            BatchNormalization(),
            Dropout(0.1),

            Dense(64, activation='relu'),
            BatchNormalization(),

            Dense(32, activation='relu'),

            Dense(self.feature_dimension, activation='tanh')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def process_player_action(self, state_vector: np.ndarray) -> Optional[np.ndarray]:
        """Обработва действие на играча и връща предсказание"""
        # Нормализация на входните данни
        normalized_state = self._normalize_state(state_vector)
        self.state_buffer.append(normalized_state)

        # Проверка за достатъчно данни
        if len(self.state_buffer) >= self.sequence_length:
            # Инкрементално обучение
            self._train_incremental()

            # Генериране на предсказание
            prediction = self._generate_prediction()

            # Оценка на увереността в предсказанието
            confidence = self._evaluate_prediction_confidence(prediction)
            self.confidence_scores.append(confidence)

            return prediction

        return None

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Нормализира входния вектор на състоянието"""
        return (state - np.mean(state)) / (np.std(state) + 1e-7)

    def _train_incremental(self):
        """Инкрементално обучение на модела"""
        if len(self.state_buffer) < self.sequence_length * 2:
            return

        # Подготовка на данните за обучение
        X = np.array([list(self.state_buffer)[i:i+self.sequence_length]
                     for i in range(len(self.state_buffer) - self.sequence_length)])
        y = np.array([list(self.state_buffer)[i]
                     for i in range(self.sequence_length, len(self.state_buffer))])

        # Обучение на модела
        history = self.model.fit(
            X, y,
            epochs=1,
            batch_size=self.config.batch_size,
            verbose=0
        )

        self.training_losses.extend(history.history['loss'])

        # Периодично обновяване на target модела
        if time.time() - self.last_update_time > self.config.target_update_frequency:
            self.target_model.set_weights(self.model.get_weights())
            self.last_update_time = time.time()

    def _generate_prediction(self) -> np.ndarray:
        """Генерира предсказание за следващото състояние"""
        recent_states = np.array(
            list(self.state_buffer)[-self.sequence_length:])
        X = recent_states.reshape(
            1, self.sequence_length, self.feature_dimension)

        # Използваме и двата модела за по-стабилни предсказания
        main_prediction = self.model.predict(X, verbose=0)[0]
        target_prediction = self.target_model.predict(X, verbose=0)[0]

        # Усредняване на предсказанията
        prediction = (main_prediction + target_prediction) / 2
        self.prediction_history.append(prediction)

        return prediction

    def _evaluate_prediction_confidence(self, prediction: np.ndarray) -> float:
        """Оценява увереността в предсказанието"""
        if len(self.prediction_history) < 2:
            return 0.5

        # Изчисляване на грешката спрямо предишни предсказания
        previous_prediction = self.prediction_history[-2]
        error = np.mean(np.abs(prediction - previous_prediction))

        # Преобразуване на грешката в увереност
        confidence = 1.0 / (1.0 + error)

        return confidence

    def get_metrics(self) -> Dict:
        """Връща текущите метрики на предсказващата система"""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'average_confidence': np.mean(self.confidence_scores),
            'recent_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'buffer_size': len(self.state_buffer),
            'predictions_made': len(self.prediction_history)
        }

    def save_model(self, path: str = 'models/predictor.h5'):
        """Запазва модела"""
        try:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str = 'models/predictor.h5'):
        """Зарежда модел"""
        try:
            self.model = tf.keras.models.load_model(path)
            self.target_model.set_weights(self.model.get_weights())
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
