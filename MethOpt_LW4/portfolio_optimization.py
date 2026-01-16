"""
Задача пошагового управления инвестиционным портфелем
Метод динамического программирования с критерием Байеса

Автор: Решение задачи оптимизации инвестиционного портфеля
Дата: 2025

Описание:
    Реализация алгоритма динамического программирования для оптимизации
    инвестиционного портфеля на трех этапах с учетом вероятностных исходов.
    Используется критерий Байеса (максимизация математического ожидания).
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Situation(Enum):
    """Типы ситуаций на рынке"""
    FAVORABLE = "благоприятная"
    NEUTRAL = "нейтральная"
    NEGATIVE = "негативная"


@dataclass
class StageData:
    """Данные для одного этапа"""
    probabilities: Dict[Situation, float]
    multipliers: Dict[Situation, Dict[str, float]]  # {situation: {asset: multiplier}}


@dataclass
class State:
    """Состояние портфеля"""
    cb1: float  # ЦБ1
    cb2: float  # ЦБ2
    deposit: float  # Депозиты
    cash: float  # Свободные средства
    
    def __hash__(self):
        return hash((round(self.cb1, 2), round(self.cb2, 2), 
                     round(self.deposit, 2), round(self.cash, 2)))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (abs(self.cb1 - other.cb1) < 0.01 and
                abs(self.cb2 - other.cb2) < 0.01 and
                abs(self.deposit - other.deposit) < 0.01 and
                abs(self.cash - other.cash) < 0.01)
    
    def total_value(self) -> float:
        """Общая стоимость портфеля"""
        return self.cb1 + self.cb2 + self.deposit + self.cash


@dataclass
class Action:
    """Действие управления"""
    buy_cb1: int = 0  # Количество пакетов ЦБ1 для покупки (отрицательное = продажа)
    buy_cb2: int = 0
    buy_deposit: int = 0


class PortfolioOptimizer:
    """Оптимизатор инвестиционного портфеля методом динамического программирования"""
    
    def __init__(self):
        # Начальные значения
        self.initial_cb1 = 100.0
        self.initial_cb2 = 800.0
        self.initial_deposit = 400.0
        self.initial_cash = 600.0
        
        # Размеры пакетов (25% от первоначальной стоимости)
        self.package_cb1 = 25.0  # 25% от 100
        self.package_cb2 = 200.0  # 25% от 800
        self.package_deposit = 100.0  # 25% от 400
        
        # Комиссии брокеров (в долях от суммы сделки)
        self.commission_cb1 = 0.04  # 4%
        self.commission_cb2 = 0.07  # 7%
        self.commission_deposit = 0.05  # 5%
        
        # Ограничения на минимальные объемы активов
        self.min_cb1 = 30.0  # Не менее 30 д.е.
        self.min_cb2 = 150.0  # Не менее 150 д.е.
        self.min_deposit = 100.0  # Не менее 100 д.е.
        
        # Данные по этапам
        self.stages = self._initialize_stages()
        
        # Кэш для мемоизации
        self.memo = {}
    
    def _initialize_stages(self) -> List[StageData]:
        """Инициализация данных по этапам"""
        stages = []
        
        # Этап 1
        stage1 = StageData(
            probabilities={
                Situation.FAVORABLE: 0.60,
                Situation.NEUTRAL: 0.30,
                Situation.NEGATIVE: 0.10
            },
            multipliers={
                Situation.FAVORABLE: {"cb1": 1.20, "cb2": 1.10, "deposit": 1.07},
                Situation.NEUTRAL: {"cb1": 1.05, "cb2": 1.02, "deposit": 1.03},
                Situation.NEGATIVE: {"cb1": 0.80, "cb2": 0.95, "deposit": 1.00}
            }
        )
        stages.append(stage1)
        
        # Этап 2
        stage2 = StageData(
            probabilities={
                Situation.FAVORABLE: 0.30,
                Situation.NEUTRAL: 0.20,
                Situation.NEGATIVE: 0.50
            },
            multipliers={
                Situation.FAVORABLE: {"cb1": 1.4, "cb2": 1.15, "deposit": 1.01},
                Situation.NEUTRAL: {"cb1": 1.05, "cb2": 1.00, "deposit": 1.00},
                Situation.NEGATIVE: {"cb1": 0.60, "cb2": 0.90, "deposit": 1.00}
            }
        )
        stages.append(stage2)
        
        # Этап 3
        stage3 = StageData(
            probabilities={
                Situation.FAVORABLE: 0.40,
                Situation.NEUTRAL: 0.40,
                Situation.NEGATIVE: 0.20
            },
            multipliers={
                Situation.FAVORABLE: {"cb1": 1.15, "cb2": 1.12, "deposit": 1.05},
                Situation.NEUTRAL: {"cb1": 1.05, "cb2": 1.01, "deposit": 1.01},
                Situation.NEGATIVE: {"cb1": 0.70, "cb2": 0.94, "deposit": 1.00}
            }
        )
        stages.append(stage3)
        
        return stages
    
    def get_initial_state(self) -> State:
        """Получить начальное состояние"""
        return State(
            cb1=self.initial_cb1,
            cb2=self.initial_cb2,
            deposit=self.initial_deposit,
            cash=self.initial_cash
        )
    
    def is_valid_action(self, state: State, action: Action) -> bool:
        """Проверка допустимости действия"""
        # Вычисляем стоимость покупки/продажи с учетом комиссий
        # При покупке: платим стоимость + комиссию
        # При продаже: получаем стоимость - комиссию
        
        if action.buy_cb1 > 0:  # Покупка
            cost_cb1 = action.buy_cb1 * self.package_cb1 * (1 + self.commission_cb1)
        elif action.buy_cb1 < 0:  # Продажа
            cost_cb1 = action.buy_cb1 * self.package_cb1 * (1 - self.commission_cb1)
        else:
            cost_cb1 = 0
        
        if action.buy_cb2 > 0:  # Покупка
            cost_cb2 = action.buy_cb2 * self.package_cb2 * (1 + self.commission_cb2)
        elif action.buy_cb2 < 0:  # Продажа
            cost_cb2 = action.buy_cb2 * self.package_cb2 * (1 - self.commission_cb2)
        else:
            cost_cb2 = 0
        
        if action.buy_deposit > 0:  # Покупка
            cost_deposit = action.buy_deposit * self.package_deposit * (1 + self.commission_deposit)
        elif action.buy_deposit < 0:  # Продажа
            cost_deposit = action.buy_deposit * self.package_deposit * (1 - self.commission_deposit)
        else:
            cost_deposit = 0
        
        total_cost = cost_cb1 + cost_cb2 + cost_deposit
        
        # Проверяем, что после действия не будет отрицательных значений
        new_cb1 = state.cb1 + action.buy_cb1 * self.package_cb1
        new_cb2 = state.cb2 + action.buy_cb2 * self.package_cb2
        new_deposit = state.deposit + action.buy_deposit * self.package_deposit
        new_cash = state.cash - total_cost  # Вычитаем, т.к. cost положительный при покупке
        
        # Проверяем ограничения:
        # 1. Нельзя иметь отрицательные активы или брать кредит
        # 2. Должны соблюдаться минимальные объемы активов
        return (new_cb1 >= self.min_cb1 and 
                new_cb2 >= self.min_cb2 and 
                new_deposit >= self.min_deposit and 
                new_cash >= 0)
    
    def apply_action(self, state: State, action: Action) -> State:
        """Применить действие к состоянию с учетом комиссий"""
        # Вычисляем стоимость покупки/продажи с учетом комиссий
        if action.buy_cb1 > 0:  # Покупка
            cost_cb1 = action.buy_cb1 * self.package_cb1 * (1 + self.commission_cb1)
        elif action.buy_cb1 < 0:  # Продажа
            cost_cb1 = action.buy_cb1 * self.package_cb1 * (1 - self.commission_cb1)
        else:
            cost_cb1 = 0
        
        if action.buy_cb2 > 0:  # Покупка
            cost_cb2 = action.buy_cb2 * self.package_cb2 * (1 + self.commission_cb2)
        elif action.buy_cb2 < 0:  # Продажа
            cost_cb2 = action.buy_cb2 * self.package_cb2 * (1 - self.commission_cb2)
        else:
            cost_cb2 = 0
        
        if action.buy_deposit > 0:  # Покупка
            cost_deposit = action.buy_deposit * self.package_deposit * (1 + self.commission_deposit)
        elif action.buy_deposit < 0:  # Продажа
            cost_deposit = action.buy_deposit * self.package_deposit * (1 - self.commission_deposit)
        else:
            cost_deposit = 0
        
        total_cost = cost_cb1 + cost_cb2 + cost_deposit
        
        return State(
            cb1=state.cb1 + action.buy_cb1 * self.package_cb1,
            cb2=state.cb2 + action.buy_cb2 * self.package_cb2,
            deposit=state.deposit + action.buy_deposit * self.package_deposit,
            cash=state.cash - total_cost  # Вычитаем общую стоимость с комиссиями
        )
    
    def apply_situation(self, state: State, stage: StageData, situation: Situation) -> State:
        """Применить ситуацию к состоянию (изменение стоимости активов)"""
        multipliers = stage.multipliers[situation]
        
        return State(
            cb1=state.cb1 * multipliers["cb1"],
            cb2=state.cb2 * multipliers["cb2"],
            deposit=state.deposit * multipliers["deposit"],
            cash=state.cash  # Наличные не меняются
        )
    
    def get_all_possible_actions(self, state: State) -> List[Action]:
        """Получить все возможные действия для состояния"""
        actions = []
        
        # Определяем максимальное количество пакетов, которые можно купить/продать
        max_buy_cb1 = int(state.cash / self.package_cb1)
        max_sell_cb1 = int(state.cb1 / self.package_cb1)
        
        max_buy_cb2 = int(state.cash / self.package_cb2)
        max_sell_cb2 = int(state.cb2 / self.package_cb2)
        
        max_buy_deposit = int(state.cash / self.package_deposit)
        max_sell_deposit = int(state.deposit / self.package_deposit)
        
        # Ограничиваем диапазон для уменьшения вычислительной сложности
        # Можно покупать/продавать от -2 до +2 пакетов каждого актива
        range_limit = 2
        
        for buy_cb1 in range(-min(range_limit, max_sell_cb1), min(range_limit, max_buy_cb1) + 1):
            for buy_cb2 in range(-min(range_limit, max_sell_cb2), min(range_limit, max_buy_cb2) + 1):
                for buy_deposit in range(-min(range_limit, max_sell_deposit), 
                                         min(range_limit, max_buy_deposit) + 1):
                    action = Action(buy_cb1=buy_cb1, buy_cb2=buy_cb2, buy_deposit=buy_deposit)
                    if self.is_valid_action(state, action):
                        actions.append(action)
        
        # Всегда добавляем действие "ничего не делать"
        actions.append(Action(0, 0, 0))
        
        return actions
    
    def bellman_value(self, stage_idx: int, state: State) -> Tuple[float, Optional[Action]]:
        """
        Рекуррентное соотношение Беллмана
        Возвращает максимальное ожидаемое значение и оптимальное действие
        """
        # Проверка кэша
        cache_key = (stage_idx, state)
        if cache_key in self.memo:
            return self.memo[cache_key]
        
        stage = self.stages[stage_idx]
        
        # Базовый случай: последний этап
        if stage_idx == len(self.stages) - 1:
            # На последнем этапе выбираем действие, максимизирующее ожидаемую стоимость
            best_value = float('-inf')
            best_action = None
            
            actions = self.get_all_possible_actions(state)
            
            for action in actions:
                new_state = self.apply_action(state, action)
                
                # Вычисляем ожидаемую стоимость после всех ситуаций
                expected_value = 0.0
                for situation, prob in stage.probabilities.items():
                    final_state = self.apply_situation(new_state, stage, situation)
                    expected_value += prob * final_state.total_value()
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            self.memo[cache_key] = (best_value, best_action)
            return best_value, best_action
        
        # Рекурсивный случай: не последний этап
        best_value = float('-inf')
        best_action = None
        
        actions = self.get_all_possible_actions(state)
        
        for action in actions:
            new_state = self.apply_action(state, action)
            
            # Вычисляем ожидаемое значение с учетом будущих этапов
            expected_value = 0.0
            for situation, prob in stage.probabilities.items():
                state_after_situation = self.apply_situation(new_state, stage, situation)
                future_value, _ = self.bellman_value(stage_idx + 1, state_after_situation)
                expected_value += prob * future_value
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        self.memo[cache_key] = (best_value, best_action)
        return best_value, best_action
    
    def solve(self) -> Tuple[float, List[Action], List[State]]:
        """
        Решить задачу оптимизации
        Возвращает: (максимальный ожидаемый доход, список оптимальных действий, список состояний)
        """
        self.memo.clear()
        initial_state = self.get_initial_state()
        
        optimal_actions = []
        states_sequence = [initial_state]
        current_state = initial_state
        
        # Проходим по всем этапам
        for stage_idx in range(len(self.stages)):
            _, action = self.bellman_value(stage_idx, current_state)
            optimal_actions.append(action)
            
            # Применяем действие
            current_state = self.apply_action(current_state, action)
            states_sequence.append(current_state)
            
            # Применяем ситуацию (для демонстрации используем ожидаемое значение)
            stage = self.stages[stage_idx]
            # Вычисляем ожидаемое состояние после ситуации
            expected_state = State(0, 0, 0, 0)
            for situation, prob in stage.probabilities.items():
                state_after = self.apply_situation(current_state, stage, situation)
                expected_state.cb1 += prob * state_after.cb1
                expected_state.cb2 += prob * state_after.cb2
                expected_state.deposit += prob * state_after.deposit
                expected_state.cash += prob * state_after.cash
            
            current_state = expected_state
        
        # Вычисляем финальное ожидаемое значение
        max_value, _ = self.bellman_value(0, initial_state)
        
        return max_value, optimal_actions, states_sequence
    
    def solve_detailed(self) -> Dict:
        """
        Решить задачу с детальной информацией о всех возможных путях
        """
        self.memo.clear()
        initial_state = self.get_initial_state()
        max_value, _ = self.bellman_value(0, initial_state)
        
        # Строим дерево решений для демонстрации
        paths = []
        
        def build_paths(state: State, stage_idx: int, path: List[Tuple[Action, Situation, State]]):
            if stage_idx >= len(self.stages):
                paths.append(path.copy())
                return
            
            _, action = self.bellman_value(stage_idx, state)
            state_after_action = self.apply_action(state, action)
            
            stage = self.stages[stage_idx]
            for situation, prob in stage.probabilities.items():
                state_after_situation = self.apply_situation(state_after_action, stage, situation)
                new_path = path + [(action, situation, state_after_situation)]
                build_paths(state_after_situation, stage_idx + 1, new_path)
        
        build_paths(initial_state, 0, [])
        
        return {
            'max_value': max_value,
            'paths': paths
        }
    
    def print_solution(self, max_value: float, actions: List[Action], states: List[State]):
        """Вывести решение задачи"""
        print("=" * 80)
        print("РЕШЕНИЕ ЗАДАЧИ ОПТИМИЗАЦИИ ИНВЕСТИЦИОННОГО ПОРТФЕЛЯ")
        print("=" * 80)
        print(f"\nМаксимальный ожидаемый доход: {max_value:.2f} д.е.")
        print(f"\nНачальное состояние:")
        print(f"  ЦБ1: {states[0].cb1:.2f} д.е.")
        print(f"  ЦБ2: {states[0].cb2:.2f} д.е.")
        print(f"  Депозиты: {states[0].deposit:.2f} д.е.")
        print(f"  Свободные средства: {states[0].cash:.2f} д.е.")
        print(f"  Общая стоимость: {states[0].total_value():.2f} д.е.")
        
        for stage_idx in range(len(self.stages)):
            print(f"\n{'=' * 80}")
            print(f"ЭТАП {stage_idx + 1}")
            print(f"{'=' * 80}")
            
            action = actions[stage_idx]
            state_before = states[stage_idx]
            state_after_action = states[stage_idx + 1] if stage_idx + 1 < len(states) else None
            
            print(f"\nОптимальное действие:")
            if action.buy_cb1 != 0:
                print(f"  {'Купить' if action.buy_cb1 > 0 else 'Продать'} {abs(action.buy_cb1)} пакет(ов) ЦБ1 "
                      f"({abs(action.buy_cb1) * self.package_cb1:.2f} д.е.)")
            if action.buy_cb2 != 0:
                print(f"  {'Купить' if action.buy_cb2 > 0 else 'Продать'} {abs(action.buy_cb2)} пакет(ов) ЦБ2 "
                      f"({abs(action.buy_cb2) * self.package_cb2:.2f} д.е.)")
            if action.buy_deposit != 0:
                print(f"  {'Купить' if action.buy_deposit > 0 else 'Продать'} {abs(action.buy_deposit)} пакет(ов) Депозитов "
                      f"({abs(action.buy_deposit) * self.package_deposit:.2f} д.е.)")
            if action.buy_cb1 == 0 and action.buy_cb2 == 0 and action.buy_deposit == 0:
                print(f"  Не предпринимать действий")
            
            print(f"\nСостояние после действия:")
            if state_after_action:
                print(f"  ЦБ1: {state_after_action.cb1:.2f} д.е.")
                print(f"  ЦБ2: {state_after_action.cb2:.2f} д.е.")
                print(f"  Депозиты: {state_after_action.deposit:.2f} д.е.")
                print(f"  Свободные средства: {state_after_action.cash:.2f} д.е.")
                print(f"  Общая стоимость: {state_after_action.total_value():.2f} д.е.")
            
            # Показываем возможные исходы после ситуаций
            stage = self.stages[stage_idx]
            print(f"\nВозможные исходы после ситуаций этапа {stage_idx + 1}:")
            if state_after_action:
                for situation, prob in stage.probabilities.items():
                    final_state = self.apply_situation(state_after_action, stage, situation)
                    print(f"  {situation.value} (вероятность {prob:.1%}): "
                          f"Общая стоимость = {final_state.total_value():.2f} д.е.")
        
        print(f"\n{'=' * 80}")


if __name__ == "__main__":
    optimizer = PortfolioOptimizer()
    max_value, actions, states = optimizer.solve()
    optimizer.print_solution(max_value, actions, states)

