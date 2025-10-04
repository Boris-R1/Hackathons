import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
from catboost import CatBoostRegressor
import sys
import argparse
import os

from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
import warnings  
warnings.filterwarnings('ignore')


class CatBoostSolution:
    def __init__(self):
        self.cv_models = {}  
        self.model_params = {
            'iterations': 2000,
            'random_state': 42,
            'verbose': 100,
            'early_stopping_rounds': 500,
            'task_type': "CPU"}  
    
    
    def load_data(self, 
                  train_candles_path: str,
                  test_candles_path: str,
                  train_news_path: str,
                  test_news_path: str):
        train_df = pd.read_csv(train_candles_path)
        train_df['begin'] = pd.to_datetime(train_df['begin'])

        test_df = pd.read_csv(test_candles_path)
        test_df['begin'] = pd.to_datetime(test_df['begin'])
        self.start_test_date_inference = test_df['begin'].min()

        # Объединяем оба теста
        self.df = pd.concat([train_df, test_df], ignore_index=True)
        self.df = self.df.sort_values(['ticker', 'begin'])

        # Loading news
        train_news = pd.read_csv(train_news_path, index_col=0)
        test_news = pd.read_csv(test_news_path)
        self.news = pd.concat([train_news, test_news], ignore_index=True)
        self.news = self.news.sort_values(['publish_date'])

        print(f"df.shape: {self.df.shape}")


    def get_company_names(self):
        """
        Получение информации о компаниях по тикерам через эндпоинт securities
        Возвращает DataFrame с полями ticker и short_name
        """
        base_url = "https://iss.moex.com/iss/securities.json"
        
        results = []
        self.tickers = self.df['ticker'].unique()

        for ticker in self.tickers:
            try:
                params = {
                    'q': ticker,
                    'lang': 'ru',
                    'securities.columns': 'secid,shortname'
                }
                
                response = requests.get(base_url, params=params)
                data = response.json()
                
                company_info = {
                    "ticker": ticker,
                    "short_name": "Не найдено"
                }
                
                if data['securities']['data']:
                    # Ищем точное совпадение по тикеру
                    for security in data['securities']['data']:
                        if security[0] == ticker:  # secid
                            company_info["short_name"] = security[1] if len(security) > 1 and security[1] else "Не найдено"
                            break
                
                results.append(company_info)
                
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "short_name": f"Ошибка: {str(e)}"
                })
        
        # Создаем DataFrame
        self.companies = pd.DataFrame(results)

    
    def fill_target_nans(self):
        """
        Filling NaNs in targets for horizons from 1 to 20 days
        """
        self.df = self.df.sort_values(['ticker', 'begin']).reset_index(drop=True)
        
        # Создаем все целевые переменные от 1 до 20 дней
        horizons = list(range(1, 21))
        
        for n_days in horizons:
            target_col = f'target_return_{n_days}d'
            self.df[target_col] = self.df.groupby('ticker')['close'].shift(-n_days) / self.df['close'] - 1
        
        print(f"Созданы целевые переменные для горизонтов: {horizons}")
        
    
    def add_exogenous_features(self):
        """
        Добавляет экзогенные признаки ИЗ ПРОШЛОГО для прогнозирования
        """
        self.df = self.df.sort_values(['ticker', 'begin']).copy()
        
        for ticker in self.df['ticker'].unique():
            mask = self.df['ticker'] == ticker
            
            # Price-based features (только прошлые данные)
            self.df.loc[mask, 'price_range'] = (self.df.loc[mask, 'high'] - self.df.loc[mask, 'low']) / self.df.loc[mask, 'close']
            self.df.loc[mask, 'price_change'] = self.df.loc[mask, 'close'] - self.df.loc[mask, 'open']
            self.df.loc[mask, 'body_ratio'] = abs(self.df.loc[mask, 'close'] - self.df.loc[mask, 'open']) / (self.df.loc[mask, 'high'] - self.df.loc[mask, 'low']).replace(0, 0.001)
            
            # Moving averages and trends (с lag)
            self.df.loc[mask, 'sma_5'] = self.df.loc[mask, 'close'].shift(1).rolling(5).mean()
            self.df.loc[mask, 'sma_20'] = self.df.loc[mask, 'close'].shift(1).rolling(20).mean()
            self.df.loc[mask, 'ema_12'] = self.df.loc[mask, 'close'].shift(1).ewm(span=12).mean()
            self.df.loc[mask, 'trend_5'] = self.df.loc[mask, 'close'] / self.df.loc[mask, 'sma_5'] - 1
            
            # Volume features (с lag)
            self.df.loc[mask, 'volume_ma_5'] = self.df.loc[mask, 'volume'].shift(1).rolling(5).mean()
            self.df.loc[mask, 'volume_ratio'] = self.df.loc[mask, 'volume'] / self.df.loc[mask, 'volume_ma_5']
            self.df.loc[mask, 'volume_price_trend'] = self.df.loc[mask, 'volume_ratio'] * self.df.loc[mask, 'trend_5']
            
            # Volatility (только прошлые доходности)
            self.df.loc[mask, 'volatility_5'] = self.df.loc[mask, 'close'].pct_change().shift(1).rolling(5).std()
            self.df.loc[mask, 'volatility_20'] = self.df.loc[mask, 'close'].pct_change().shift(1).rolling(20).std()
            
            # Momentum indicators (только прошлые данные)
            self.df.loc[mask, 'momentum_5'] = self.df.loc[mask, 'close'] / self.df.loc[mask, 'close'].shift(5) - 1
            self.df.loc[mask, 'momentum_20'] = self.df.loc[mask, 'close'] / self.df.loc[mask, 'close'].shift(20) - 1
            
            # RSI (только прошлые данные)
            returns = self.df.loc[mask, 'close'].diff().shift(1)
            gain = returns.clip(lower=0).rolling(14).mean()
            loss = returns.clip(upper=0).abs().rolling(14).mean()
            self.df.loc[mask, 'rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))
            
            # Support/resistance levels (только прошлые данные)
            self.df.loc[mask, 'resistance_20'] = self.df.loc[mask, 'high'].shift(1).rolling(20).max()
            self.df.loc[mask, 'support_20'] = self.df.loc[mask, 'low'].shift(1).rolling(20).min()
            self.df.loc[mask, 'dist_to_resistance'] = (self.df.loc[mask, 'resistance_20'] - self.df.loc[mask, 'close']) / self.df.loc[mask, 'close']
            self.df.loc[mask, 'dist_to_support'] = (self.df.loc[mask, 'close'] - self.df.loc[mask, 'support_20']) / self.df.loc[mask, 'close']
        
        # Time-based features (без data leakage)
        self.df['day_of_week'] = self.df['begin'].dt.dayofweek
        self.df['month'] = self.df['begin'].dt.month
        self.df['quarter'] = self.df['begin'].dt.quarter
        self.df['year'] = self.df['begin'].dt.year
        self.df['is_month_end'] = self.df['begin'].dt.is_month_end.astype(int)


    def add_advanced_features(self):
        """
        Добавляет расширенные технические индикаторы ИЗ ПРОШЛОГО
        """
        self.df = self.df.sort_values(['ticker', 'begin']).copy()
        
        for ticker in self.df['ticker'].unique():
            mask = self.df['ticker'] == ticker
            
            # 1. ATR_14 (Average True Range) - волатильность (СДВИГ!)
            high_low = self.df.loc[mask, 'high'] - self.df.loc[mask, 'low']
            high_close = abs(self.df.loc[mask, 'high'] - self.df.loc[mask, 'close'].shift(1))
            low_close = abs(self.df.loc[mask, 'low'] - self.df.loc[mask, 'close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df.loc[mask, 'atr_14'] = true_range.shift(1).rolling(14).mean()  # ← shift(1)
            
            # 2. MACD (Moving Average Convergence Divergence) - тренд (СДВИГ!)
            ema_12 = self.df.loc[mask, 'close'].shift(1).ewm(span=12).mean()  # ← shift(1)
            ema_26 = self.df.loc[mask, 'close'].shift(1).ewm(span=26).mean()  # ← shift(1)
            self.df.loc[mask, 'macd'] = ema_12 - ema_26
            self.df.loc[mask, 'macd_signal'] = self.df.loc[mask, 'macd'].ewm(span=9).mean()
            self.df.loc[mask, 'macd_histogram'] = self.df.loc[mask, 'macd'] - self.df.loc[mask, 'macd_signal']
            
            # 3. Stochastic Oscillator - перекупленность/перепроданность (СДВИГ!)
            low_14 = self.df.loc[mask, 'low'].shift(1).rolling(14).min()  # ← shift(1)
            high_14 = self.df.loc[mask, 'high'].shift(1).rolling(14).max()  # ← shift(1)
            self.df.loc[mask, 'stochastic_k'] = 100 * (self.df.loc[mask, 'close'] - low_14) / (high_14 - low_14).replace(0, 0.001)
            self.df.loc[mask, 'stochastic_d'] = self.df.loc[mask, 'stochastic_k'].rolling(3).mean()
            
            # 4. Volume Surge - аномалии объема (СДВИГ!)
            volume_ma_20 = self.df.loc[mask, 'volume'].shift(1).rolling(20).mean()  # ← shift(1)
            volume_std_20 = self.df.loc[mask, 'volume'].shift(1).rolling(20).std()  # ← shift(1)
            self.df.loc[mask, 'volume_surge'] = (self.df.loc[mask, 'volume'] - volume_ma_20) / volume_std_20.replace(0, 0.001)
            
            # 5. Price Z-Score - отклонение от нормы (СДВИГ!)
            price_ma_20 = self.df.loc[mask, 'close'].shift(1).rolling(20).mean()  # ← shift(1)
            price_std_20 = self.df.loc[mask, 'close'].shift(1).rolling(20).std()  # ← shift(1)
            self.df.loc[mask, 'price_zscore'] = (self.df.loc[mask, 'close'] - price_ma_20) / price_std_20.replace(0, 0.001)
            
            # 6. Skewness и Kurtosis распределения доходностей (СДВИГ!)
            returns = self.df.loc[mask, 'close'].pct_change().shift(1)  # ← shift(1)
            self.df.loc[mask, 'returns_skew_10'] = returns.rolling(10).skew()
            self.df.loc[mask, 'returns_kurt_10'] = returns.rolling(10).kurt()
            
            # 7. Дополнительные волатильностные признаки (СДВИГ!)
            self.df.loc[mask, 'volatility_ratio'] = (
                self.df.loc[mask, 'close'].pct_change().shift(1).rolling(5).std() /  # ← shift(1)
                self.df.loc[mask, 'close'].pct_change().shift(1).rolling(20).std().replace(0, 0.001)  # ← shift(1)
            )
        
        # 8. Циклические временные признаки (без data leakage)
        self.df['day_of_year'] = self.df['begin'].dt.dayofyear
        self.df['week_of_year'] = self.df['begin'].dt.isocalendar().week
        self.df['is_quarter_start'] = self.df['begin'].dt.is_quarter_start.astype(int)
        
        # Циклическое кодирование (без data leakage)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # 9. Дополнительные ценовые отношения (без data leakage)
        self.df['close_open_ratio'] = self.df['close'] / self.df['open']
        self.df['high_low_ratio'] = self.df['high'] / self.df['low']    


    def train_test_split_n_days(self, start_test_date, end_test_date, n_days=20):
        """
        Разделяет датафрейм на train и test по заданной дате
        Удаляет n последних дат для каждого тикера в train
        
        Parameters:
        -----------
        start_test_date : str или datetime
            Начальная дата тестового периода
        end_test_date : str или datetime
            Конечная дата тестового периода  
        n_days : int, default=20
            Количество дней для предсказания (от 1 до 20)
        """
        # Проверяем корректность n_days
        if not 1 <= n_days <= 20:
            raise ValueError("n_days должен быть в диапазоне от 1 до 20")
        
        # Разделяем по дате
        self.train_df_n = self.df[self.df['begin'] < start_test_date].copy()
        self.test_df_n = self.df[(self.df['begin'] >= start_test_date) & 
                                (self.df['begin'] <= end_test_date)].copy()
        
        # Удаляем n последних дат для каждого тикера в train
        def remove_last_n(group):
            return group.iloc[:-n_days] if len(group) > n_days else pd.DataFrame()
        
        self.train_df_n = self.train_df_n.groupby('ticker').apply(remove_last_n).reset_index(drop=True)
        
        # Удаляем строки с NaN в соответствующем целевом столбце
        target_column = f'target_return_{n_days}d'
        self.test_df_n = self.test_df_n.dropna(subset=[target_column])
        
        print(f"Prediction horizon: {n_days} days")
        print(f"Train size: {len(self.train_df_n)}")
        print(f"Test size: {len(self.test_df_n)}")
        print(f"Train date range: {self.train_df_n['begin'].min()} - {self.train_df_n['begin'].max()}")
        print(f"Test date range: {self.test_df_n['begin'].min()} - {self.test_df_n['begin'].max()}")
        print('-' * 20)
        
        # Сохраняем n_days как атрибут для дальнейшего использования
        self.n_days = n_days

    
    def specify_features(self):
        # Подготовка фичей (исключаем служебные колонки и таргеты)
        exclude_cols = ['begin', 'target_return_1d', 'target_direction_1d', 
                    'target_return_20d', 'target_direction_20d']
        
        harmful_features = ['resistance_20', 'macd', 'sma_20', 'momentum_20']
        
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols and col not in harmful_features]


    def predict_return_n_days(self, n_days=20, target_col=None, i=None):
        """
        Обучает модель на train_df и предсказывает target_return_{n_days}d на test_df
        
        Parameters:
        -----------
        n_days : int, default=20
            Количество дней для предсказания (от 1 до 20)
        target_col : str, optional
            Название целевой колонки. Если None, генерируется автоматически
        i : int, optional
            Индекс для кросс-валидации
        """
        # Проверяем корректность n_days
        if not 1 <= n_days <= 20:
            raise ValueError("n_days должен быть в диапазоне от 1 до 20")
        
        # Определяем целевой столбец
        if target_col is None:
            target_col = f'target_return_{n_days}d'
        
        # Получаем соответствующие train и test датафреймы
        train_df = self.train_df_n
        test_df = self.test_df_n
        
        # Данные для обучения
        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col]
        
        # Данные для теста
        X_test = test_df[self.feature_cols]
        
        # Определяем категориальные фичи
        categorical_features = [col for col in self.feature_cols if 
                            train_df[col].dtype == 'object' or
                            col in ['ticker']]
        
        # Создаем и обучаем модель
        model = CatBoostRegressor(
            cat_features=categorical_features, 
            **self.model_params, 
            eval_metric='MAE'
        )
        
        # Обучение
        model.fit(X_train, y_train, eval_set=(X_test, test_df[target_col]))
        
        # Сохраняем модель в соответствующий атрибут
        model_attr_name = f'model_ret_{n_days}d'
        setattr(self, model_attr_name, model)
        
        # Сохранение для кросс-валидации
        if i is not None:
            self.cv_models[f'model_ret_{n_days}d_cv_{i}'] = model
        
        # Предсказание
        pred_col_name = f'pred_return_{n_days}d'
        test_df[pred_col_name] = model.predict(X_test)

        mae = mean_absolute_error(test_df[target_col], test_df[pred_col_name])        
        print(f"Обучена модель для предсказания на {n_days} дней")
        print(f"Качество на тесте (MAE): {mae:.4f}")


    def predict_cv_with_metrics(self, cv=5, test_window=22, horizons=None):
        """
        Кросс-валидация для предсказаний в диапазоне от 1 до 20 дней
        
        Parameters:
        -----------
        cv : int, default=5
            Количество фолдов
        test_window : int, default=22
            Размер тестового окна в днях
        horizons : list of int, optional
            Список горизонтов предсказания. Если None, используется [1, 2, ..., 20]
        """
        if horizons is None:
            horizons = list(range(1, 21))
        
        # Инициализация результатов
        metrics = [f'mae_{n_days}d' for n_days in horizons]
        results = {metric: [] for metric in metrics}
        dates = []
        
        end_date = pd.to_datetime(self.df['begin'].max())
        
        for i in range(cv):
            if i == 0:
                start_date = end_date - pd.Timedelta(days=46)
            else:
                start_date -= pd.Timedelta(days=test_window)
            
            print(f"Фолд {i+1}/{cv}: {start_date.date()} - {end_date.date()}")
            
            # Словарь для хранения MAE по каждому объекту для каждого горизонта
            object_mae_by_horizon = {n_days: [] for n_days in horizons}
            
            for n_days in horizons:
                # Train/test split и предсказание для каждого горизонта
                self.train_test_split_n_days(start_date, end_date, n_days=n_days)
                self.predict_return_n_days(n_days=n_days, i=i)
                
                # Получаем соответствующий test_df
                test_df = self.test_df_n
                
                # Вычисляем MAE для каждого объекта в текущем горизонте
                target_col = f'target_return_{n_days}d'
                pred_col = f'pred_return_{n_days}d'
                
                # MAE для каждого объекта (строки)
                object_mae = np.abs(test_df[target_col] - test_df[pred_col]).values
                object_mae_by_horizon[n_days].extend(object_mae)
                
                print(f"  Horizon {n_days}d: объектов = {len(object_mae)}, средний MAE = {np.mean(object_mae):.4f}")
            
            # Усредняем MAE по всем объектам для каждого горизонта в текущем фолде
            fold_metrics = {}
            for n_days in horizons:
                mae_fold = np.mean(object_mae_by_horizon[n_days])
                fold_metrics[f'mae_{n_days}d'] = mae_fold
                print(f"  Horizon {n_days}d: итоговый MAE по фолду = {mae_fold:.4f}")
            
            # Сохраняем метрики фолда
            for metric in metrics:
                results[metric].append(fold_metrics[metric])
            
            dates.append((start_date, end_date))
            
            if i == 0:
                end_date -= pd.Timedelta(days=46)
            else:
                end_date -= pd.Timedelta(days=test_window)
        
        # Вычисляем средние метрики по всем фолдам
        avg_metrics = {}
        for metric in metrics:
            avg_value = round(np.mean(results[metric]), 4)
            avg_metrics[metric] = avg_value
            setattr(self, metric, avg_value)
        
        # Создаем DataFrame с результатами
        self.cv_results = pd.DataFrame({
            'fold': range(cv),
            'start_test_date': [d[0] for d in dates],
            'end_test_date': [d[1] for d in dates],
            **{metric: results[metric] for metric in metrics}
        })

        # computing average metrics        
        mae_cols = [col for col in self.cv_results.columns if 'mae_' in col]
        self.cv_results.loc['avg', :] = self.cv_results[mae_cols].mean(axis=0)
        self.cv_results.loc['avg', 'fold'] = 'avg'

        mae_cols = [col for col in self.cv_results.columns if 'mae_' in col]
        self.cv_results['mae_avg'] = self.cv_results[mae_cols].mean(axis=1)

        self.cv_results = round(self.cv_results, 4)


    def train_test_split_inference(self):
        """
        Разделяет датафрейм на train и test по заданной дате для inference

        """
        
        # Разделяем по дате
        self.train_df_inference = self.df[self.df['begin'] < self.start_test_date_inference].copy()
        self.test_df_inference = self.df[self.df['begin'] >= self.start_test_date_inference].copy()


    def predict_return_inference(self, n_days_range=range(1, 21)):
        """
        Обучает модели на train_df_inference и предсказывает target_return_{n_days}d 
        на test_df_inference для диапазона дней
        
        Parameters:
        -----------
        n_days_range : range, default=range(1, 21)
            Диапазон дней для предсказания (от 1 до 20)
        """
        
        # Получаем соответствующие train и test датафреймы
        train_df = self.train_df_inference.copy()
        test_df = self.test_df_inference
        
        # Определяем категориальные фичи
        categorical_features = [col for col in self.feature_cols if 
                            train_df[col].dtype == 'object' or
                            col in ['ticker']]
        
        # Обучаем модели и делаем предсказания для каждого n_days
        for n_days in n_days_range:
            target_col = f'target_return_{n_days}d'
            pred_col_name = f'pred_return_{n_days}d'
            
            print(f"Обучение модели для предсказания на {n_days} дней...")
            
            # Удаляем n_days последних дат для каждого тикера в train
            def remove_last_n(group):
                return group.iloc[:-n_days] if len(group) > n_days else pd.DataFrame()
            
            train_df_filtered = train_df.groupby('ticker').apply(remove_last_n).reset_index(drop=True)
            
            # Проверяем, что после фильтрации остались данные
            if len(train_df_filtered) == 0:
                print(f"Предупреждение: после удаления {n_days} последних дат train_df пуст. Пропускаем n_days={n_days}")
                continue
            
            # Данные для обучения
            X_train = train_df_filtered[self.feature_cols]
            y_train = train_df_filtered[target_col]
            
            # Данные для теста
            X_test = test_df[self.feature_cols]
            
            # Создаем и обучаем модель
            model = CatBoostRegressor(
                cat_features=categorical_features, 
                **self.model_params
            )
            
            # Обучение
            model.fit(X_train, y_train)

            # Предсказание
            test_df[pred_col_name] = model.predict(X_test)
            
            # Оценка качества
            print(f"Предикты для {n_days} дней получены.")
        
        print("-" * 50)
    

    def log_results_to_mlflow(self, experiment_name="Finam hackathon", description=None):
        """
        Сохраняет результаты и параметры в MLflow
        """
        # Connecting
        mlflow.set_tracking_uri("http://localhost:5000")
        print("Подключено к MLflow:", mlflow.get_tracking_uri())

        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"catboost_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"):
            if description:
                mlflow.set_tag("description", description)

            # Логируем метрики
            mlflow.log_metric("mae_1d", float(self.cv_results.loc['avg', 'mae_1d']))
            mlflow.log_metric("mae_20d", float(self.cv_results.loc['avg', 'mae_20d']))
            mlflow.log_metric("mae_avg", float(self.cv_results.loc['avg', 'mae_avg']))

            # Логируем метрики для каждого фолда CV
            if hasattr(self, 'cv_results'):
                self.cv_results.to_csv("cv_results.csv", index=False)
                mlflow.log_artifact("cv_results.csv")
            
            # Логируем параметры моделей
            mlflow.log_param("model_params", self.model_params)
            
            # Логируем фичи
            mlflow.log_param('n_features', len(self.feature_cols))
            mlflow.log_param('features', self.feature_cols)

        print('Results logged')
        print('-'*20)
   

    def create_submission_file(self, output_path='submission.csv'):
        """
        Формирует файл submission.csv с предиктами из test_df_inference
        
        Parameters:
        -----------
        output_path : str, default='submission.csv'
            Путь для сохранения файла submission.csv
        """
        # Проверяем, что test_df_inference существует и содержит предикты
        if not hasattr(self, 'test_df_inference'):
            print("Ошибка: test_df_inference не найден")
            return
        
        # Получаем все столбцы с предиктами и сортируем их по числовому значению
        pred_cols = sorted([col for col in self.test_df_inference.columns 
                        if col.startswith('pred_return_')], 
                        key=lambda x: int(x.split('_')[-1].replace('d', '')))
        
        # Создаем mapping для переименования колонок
        column_mapping = {}
        new_pred_cols = []
        
        for i, col in enumerate(pred_cols, 1):
            new_name = f'p{i}'
            column_mapping[col] = new_name
            new_pred_cols.append(new_name)
        
        # Создаем submission датафрейм только с тикером и предиктами
        submission_cols = ['ticker'] + pred_cols
        self.submission_df = self.test_df_inference[submission_cols].copy()
        
        # Переименовываем колонки
        self.submission_df = self.submission_df.rename(columns=column_mapping)
        
        # Убеждаемся, что колонки расположены в правильном порядке: ticker, p1, p2, ..., p20
        final_cols = ['ticker'] + new_pred_cols
        self.submission_df = self.submission_df[final_cols]
        
        # Сортируем по ticker
        self.submission_df = self.submission_df.sort_values(['ticker']).reset_index(drop=True)
        
        # Сохраняем в CSV
        self.submission_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run CatBoostSolution pipeline")
    parser.add_argument('--train-candles-path', default=".//Finam//data_v2//candles.csv",
    help='path to train candles CSV')
    parser.add_argument('--test-candles-path', default=".//Finam//data_v2//candles_2.csv",
    help='path to test candles CSV')
    parser.add_argument('--train-news-path', default=".//Finam//data_v2//news.csv",
    help='path to train news CSV')
    parser.add_argument('--test-news-path', default=".//Finam//data_v2//news_2.csv",
    help='path to test news CSV')
    parser.add_argument('--output-path', default=".//Finam//submission.csv",
    help='output submission CSV path')


    args = parser.parse_args()

    cb = CatBoostSolution()

    cb.load_data(args.train_candles_path, args.test_candles_path, args.train_news_path, args.test_news_path)
    cb.fill_target_nans()
    cb.add_exogenous_features()
    cb.add_advanced_features()
    cb.specify_features()

    cb.train_test_split_inference()
    cb.predict_return_inference(n_days_range=range(1, 21))
    cb.create_submission_file(args.output_path)

if __name__ == '__main__':
    main()