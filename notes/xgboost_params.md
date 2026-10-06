# Финальные гиперпараметры XGBoost

## Модель

Одна глобальная панельная модель. Строка = «регион-год». Обучение на всех 85 регионах одновременно. Целевая переменная — `investments` (инвестиции в основной капитал, млн руб.), с логарифмированием `log1p`.

## Гиперпараметры

| Параметр | Значение |
|---|---|
| objective | reg:squarederror |
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 3 |
| reg_alpha | 0.0 |
| reg_lambda | 1.0 |
| random_state | 42 |

## Список признаков (22 штуки)

Лаги таргета:
- investments_lag1
- investments_lag2
- investments_lag3

Лаги внешних показателей:
- retail_lag1, retail_lag2, retail_lag3
- income_lag1, income_lag2, income_lag3
- unemployment_lag1, unemployment_lag2, unemployment_lag3
- housing_lag1, housing_lag2, housing_lag3
- population_lag1_lag1, population_lag1_lag2, population_lag1_lag3

Скользящие статистики:
- investments_roll_mean_3
- investments_roll_mean_5
- investments_roll_std_3

Логарифм лага:
- investments_log_lag1

## Валидация и разбиение

**Схема:** Expanding Window TimeSeriesSplit, 4 фолда внутри периода 2002–2020.

| Фолд | Обучение | Валидация |
|---|---|---|
| 1 | ≤ 2014 | 2015–2016 |
| 2 | ≤ 2016 | 2017–2018 |
| 3 | ≤ 2018 | 2019 |
| 4 | ≤ 2019 | 2020 |

**Train:** 2002–2020 (1136 строк)  
**Test:** 2021–2024 (340 строк, 85 регионов × 4 года)  
**Early stopping:** не использовался — фиксация `n_estimators=300` по результатам кросс-валидации.

## Выбор гиперпараметров

Из сетки 4 конфигураций (`n_estimators ∈ {200, 300}`, `max_depth ∈ {3, 4}`, `learning_rate ∈ {0.03, 0.05}`) выбрана финальная по минимальному sMAPE на валидации:

| Конфигурация | n_estimators | max_depth | learning_rate | sMAPE |
|---|---:|---:|---:|---:|
| 1 | 200 | 3 | 0.03 | 13.66% |
| 2 | 300 | 3 | 0.03 | 13.49% |
| 3 | 300 | 4 | 0.03 | 13.45% |
| 4 | 300 | 4 | 0.05 | 13.51% |

**Финальная:** конфигурация 3 — `n_estimators=300, max_depth=4, learning_rate=0.03`.