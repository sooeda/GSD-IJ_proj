"""
Unified GSD-IJ AHP Architecture
Вариант A: Честная замена центра — все вычисления относительно центра

Архитектура позволяет переключаться между методами расчёта центра:
- WGMM (взвешенное геометрическое среднее)
- RowSum (сумма строк)
- TropicalCenter (тропический log-Чебышевский центр)

Все формулы и вычисления остаются идентичными — меняется только способ получения центра.
"""

import numpy as np
import pulp
from typing import List, Tuple, Dict, Optional, Any, Literal
import warnings
import json


class GSDIJ_AHP:
    """
    GSD-IJ AHP с поддержкой разных методов расчёта центра (center_method).
    """

    def __init__(self, debug_mode: bool = False, center_method: Literal["wgmm", "rowsum", "tropical"] = "wgmm", tropical_variant: Literal["best", "worst"] = "best"):
        """
        Args:
            debug_mode: Выводить ли отладочную информацию
            center_method: Метод расчёта центра:
                - "wgmm": взвешенное геометрическое среднее
                - "rowsum": сумма строк
                - "tropical": тропический log-Чебышевский центр
        """
        self.n = 0  # Количество критериев
        self.m = 0  # Количество экспертов
        self.individual_matrices: List[np.ndarray] = []  # Индивидуальные матрицы
        self.alpha_weights: List[float] = []  # Веса экспертов
        self.criteria_names: List[str] = []  # Названия критериев
        self.dm_ids: List[str] = []  # Идентификаторы экспертов
        self.lambda_opt: float = 0.0  # Оптимальное значение лямбды
        self.group_interval_matrix: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.group_weights: Optional[np.ndarray] = None
        self.debug_mode = debug_mode
        
        # === НОВОЕ: центр и метод расчёта ===
        self.center_method = center_method.lower()
        self.center_matrix: Optional[np.ndarray] = None  # Вычисленный центр
        self.tropical_mu: Optional[float] = None  # Используется для tropical метода
        tropical_variant: Literal["best", "worst"] = "best" 

        if self.center_method not in ["wgmm", "rowsum", "tropical"]:
            raise ValueError(f"Unknown center_method: {center_method}")
        
        self.tropical_variant = tropical_variant.lower()
    
        if self.tropical_variant not in ["best", "worst"]:
            raise ValueError(f"Unknown tropical_variant: {tropical_variant}")
    # ==================== LOADING ====================

    def load_from_json(self, filepath: str) -> None:
        """Загрузка данных из JSON файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Загрузка критериев
            if 'criteria' in data:
                if isinstance(data['criteria'], list) and len(data['criteria']) > 0:
                    if isinstance(data['criteria'][0], dict):
                        self.criteria_names = [criterion['name'] for criterion in data['criteria']]
                    else:
                        self.criteria_names = data['criteria']
                else:
                    raise ValueError("Критерии не заданы или пусты")
                self.n = len(self.criteria_names)
            else:
                raise ValueError("Не найдены названия критериев в JSON файле")

            if 'dms' not in data:
                raise ValueError("Не найдены данные экспертов в JSON файле")

            self.individual_matrices = []
            self.dm_ids = []

            for dm_data in data['dms']:
                dm_id = dm_data.get('id', f'DM{len(self.dm_ids) + 1}')
                self.dm_ids.append(dm_id)

                if 'pairwise_comparisons' in dm_data:
                    matrix = np.array(dm_data['pairwise_comparisons'], dtype=float)
                    if matrix.shape != (self.n, self.n):
                        raise ValueError(
                            f"Матрица эксперта {dm_id} имеет неверный размер: {matrix.shape}, "
                            f"ожидается ({self.n}, {self.n})"
                        )
                    self.individual_matrices.append(matrix)
                elif 'matrix' in dm_data:
                    matrix = np.array(dm_data['matrix'], dtype=float)
                    if matrix.shape != (self.n, self.n):
                        raise ValueError(f"Матрица эксперта {dm_id} имеет неверный размер")
                    self.individual_matrices.append(matrix)
                else:
                    raise ValueError(f"Не найдена матрица сравнений для эксперта {dm_id}")

            self.m = len(self.individual_matrices)

            # Проверка на положительность
            for k, A in enumerate(self.individual_matrices):
                if np.any(A <= 0):
                    raise ValueError(
                        f"Матрица эксперта {self.dm_ids[k]} содержит неположительные значения"
                    )

            # Загрузка весов экспертов
            if 'parameters' in data:
                params = data['parameters']
                if 'alpha_weights' in params:
                    weights = params['alpha_weights']
                    if len(weights) != self.m:
                        raise ValueError(
                            f"Количество весов экспертов ({len(weights)}) не совпадает "
                            f"с количеством экспертов ({self.m})"
                        )
                    self.alpha_weights = [float(w) for w in weights]
                else:
                    self.alpha_weights = [1.0 / self.m] * self.m
            else:
                self.alpha_weights = [1.0 / self.m] * self.m

            # Нормализация весов
            total_weight = sum(self.alpha_weights)
            if total_weight != 1.0:
                self.alpha_weights = [w / total_weight for w in self.alpha_weights]

            if self.debug_mode:
                print(f"Успешно загружено из {filepath}")
                print(f" Критериев: {self.n}")
                print(f" Экспертов: {self.m}")
                print(f" Метод расчёта центра: {self.center_method}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {filepath} не найден")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка чтения JSON файла: {e}")
        except Exception as e:
            raise ValueError(f"Ошибка загрузки данных: {e}")

    # ==================== ЦЕНТР: ТРИ МЕТОДА ====================

    def _compute_wgmm_center(self) -> np.ndarray:
        """
        Взвешенное геометрическое среднее (Weighted Geometric Mean Matrix).
        """
        wgmm_matrix = np.ones((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                product = 1.0
                for k in range(self.m):
                    product *= self.individual_matrices[k][i, j] ** self.alpha_weights[k]
                wgmm_matrix[i, j] = product

        return wgmm_matrix

    def _compute_rowsum_center(self) -> np.ndarray:
        """
        Метод суммы строк (Row Sum Method).
        """
        if self.m == 1:
            return self.individual_matrices[0].copy()

        # 1. Для каждого эксперта вычисляем веса методом строчных сумм
        expert_weights = []

        for k in range(self.m):
            matrix_k = self.individual_matrices[k]

            # Вычисляем суммы строк
            row_sums = np.zeros(self.n)
            for i in range(self.n):
                row_sum = 0.0
                for j in range(self.n):
                    row_sum += matrix_k[i, j]
                row_sums[i] = row_sum

            # Нормализуем суммы строк для получения вектора весов
            total_sum = np.sum(row_sums)
            weights_k = row_sums / total_sum
            expert_weights.append(weights_k)

        # 2. Вычисляем групповые веса как взвешенное среднее индивидуальных весов
        group_weights = np.zeros(self.n)
        for i in range(self.n):
            weighted_sum = 0.0
            for k in range(self.m):
                weighted_sum += self.alpha_weights[k] * expert_weights[k][i]
            group_weights[i] = weighted_sum

        # Нормализуем групповые веса
        group_weights = group_weights / np.sum(group_weights)

        # 3. Строим групповую матрицу из групповых весов
        group_matrix = np.ones((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if group_weights[j] > 0:
                        group_matrix[i, j] = group_weights[i] / group_weights[j]
                    else:
                        group_matrix[i, j] = 1.0

        return group_matrix

    def _compute_tropical_center(self) -> Tuple[np.ndarray, float]:
        """
        Тропический log-Чебышевский центр (Best и параметр му).
        """
        n = self.n
        m = self.m

        # 1) B_ij = max_k (w_k * a_ij^(k))
        B = np.zeros((n, n), dtype=float)
        np.fill_diagonal(B, 1.0)

        for k in range(m):
            A = np.asarray(self.individual_matrices[k], dtype=float)
            w = float(self.alpha_weights[k])
            if w <= 0:
                raise ValueError(f"Вес эксперта {k} должен быть > 0, получено {w}")
            if A.shape != (n, n):
                raise ValueError(f"Матрица эксперта {k} имеет размер {A.shape}, ожидалось {(n, n)}")
            if np.any(A <= 0):
                i0, j0 = np.argwhere(A <= 0)[0]
                raise ValueError(f"a_ij должно быть > 0 (нашли {A[i0, j0]} у эксперта {k}, пара {(i0, j0)})")

            W = w * A
            np.fill_diagonal(W, 1.0)
            B = np.maximum(B, W)

        # 2) μ и S = (μ^{-1}B)^*
        mu = self.trop_spectral_radius_trace_formula(B)
        S = self._kleene_star(B / mu)

        if self.tropical_variant == "best":
            x = self._best_diff_vector_from_star(S)
        else:  # worst
            x = self._worst_diff_vector_from_matrix(B, mu, S)


        # 3) Best
        # x_best = self._best_diff_vector_from_star(S)
        # x_best = x_best / x_best[0]

        x = x / x[0]
        C = x.reshape(-1, 1) / x.reshape(1, -1)
        
        self.tropical_mu = mu
        return C, mu

    def compute_center_matrix(self) -> np.ndarray:
        """
        Вычисляет матрицу центра в зависимости от center_method.
        Результат сохраняется в self.center_matrix.
        """
        if self.center_method == "wgmm":
            self.center_matrix = self._compute_wgmm_center()
        elif self.center_method == "rowsum":
            self.center_matrix = self._compute_rowsum_center()
        elif self.center_method == "tropical":
            self.center_matrix, _ = self._compute_tropical_center()
        else:
            raise ValueError(f"Unknown center_method: {self.center_method}")

        return self.center_matrix

    # ==================== DISPERSION (СКО) ====================

    def compute_geometric_std_matrix(self, center_matrix: np.ndarray) -> np.ndarray:
        """
        Вычисление матрицы геометрических стандартных отклонений относительно центра.
        Формула одинакова для всех методов расчёта центра!
        """
        std_matrix = np.ones((self.n, self.n))
        sum_alpha_sq = sum(a ** 2 for a in self.alpha_weights)
        denominator = max(1.0 - sum_alpha_sq, 1e-10)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                sum_sq = 0.0
                for k in range(self.m):
                    ratio = self.individual_matrices[k][i, j] / center_matrix[i, j]
                    log_ratio = np.log(ratio)
                    sum_sq += self.alpha_weights[k] * (log_ratio ** 2)

                std_matrix[i, j] = np.exp(np.sqrt(sum_sq / denominator))

        return std_matrix

    # ==================== ИНТЕРВАЛЫ ====================

    def compute_group_interval_matrix(self, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление интервальной групповой матрицы.
        Формула одинакова — меняется только центр!
        """
        if self.center_matrix is None:
            self.compute_center_matrix()

        center_matrix = self.center_matrix
        std_matrix = self.compute_geometric_std_matrix(center_matrix)

        lower_matrix = np.ones((self.n, self.n))
        upper_matrix = np.ones((self.n, self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                center_val = center_matrix[i, j]
                std_val = std_matrix[i, j]

                lower_matrix[i, j] = center_val / (std_val ** lambda_val)
                upper_matrix[i, j] = center_val * (std_val ** lambda_val)

                lower_matrix[j, i] = 1.0 / upper_matrix[i, j]
                upper_matrix[j, i] = 1.0 / lower_matrix[i, j]

        return lower_matrix, upper_matrix

    # ==================== ИНДЕКСЫ ====================

    def compute_indeterminacy_index(self, lower_matrix: np.ndarray, upper_matrix: np.ndarray) -> float:
        """Вычисление индекса неопределённости интервальной матрицы."""
        if self.n < 2:
            return 1.0

        product = 1.0
        count = 0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if lower_matrix[i, j] > 0:
                    ratio = upper_matrix[i, j] / lower_matrix[i, j]
                    product *= ratio
                    count += 1

        if count > 0:
            return product ** (2.0 / (self.n * (self.n - 1)))
        return 1.0

    def compute_group_satisfaction_index(self, lower_matrix: np.ndarray, upper_matrix: np.ndarray) -> float:
        """Вычисление индекса групповой удовлетворённости (GSI)."""
        if self.n < 2:
            return 1.0

        total_satisfaction = 0.0
        total_comparisons = self.n * (self.n - 1) // 2

        for k in range(self.m):
            count_satisfied = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    individual_value = self.individual_matrices[k][i, j]
                    if lower_matrix[i, j] <= individual_value <= upper_matrix[i, j]:
                        count_satisfied += 1

            satisfaction_k = count_satisfied / total_comparisons
            total_satisfaction += self.alpha_weights[k] * satisfaction_k

        return total_satisfaction

    # ==================== ОПТИМИЗАЦИЯ ЛЯМБДЫ ====================

    def solve_model_1(self, t: float = 3.0, s: float = 0.5, n_points: int = 1001) -> float:
        """Оптимизация параметра лямбды."""
        lambdas = np.linspace(0.0, 1.0, n_points)
        feasible_lambdas = []

        for lambda_val in lambdas:
            lower, upper = self.compute_group_interval_matrix(lambda_val)
            U = self.compute_indeterminacy_index(lower, upper)
            GSI = self.compute_group_satisfaction_index(lower, upper)

            if U <= t and GSI >= s:
                feasible_lambdas.append(lambda_val)

        if feasible_lambdas:
            return min(feasible_lambdas)

        if self.debug_mode:
            print("Model 1 не имеет решения. Ищем лямбду, при котором U = t")

        # Бинарный поиск
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lower, upper = self.compute_group_interval_matrix(mid)
            U = self.compute_indeterminacy_index(lower, upper)

            if U < t:
                lo = mid
            else:
                hi = mid

        lambda_fallback = (lo + hi) / 2

        if self.debug_mode:
            lower, upper = self.compute_group_interval_matrix(lambda_fallback)
            U_final = self.compute_indeterminacy_index(lower, upper)
            GSI_final = self.compute_group_satisfaction_index(lower, upper)
            print(f"λ = {lambda_fallback:.4f}, U = {U_final:.4f}, GSI = {GSI_final:.4f}")

        return lambda_fallback

    # ==================== ВЕСА (FPP) ====================

    def solve_model_2_fpp(self, lower_matrix: np.ndarray, upper_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """Решение Model 2: метод нечёткого программирования предпочтений (FPP)."""
        prob = pulp.LpProblem("FPP_Weights", pulp.LpMaximize)

        c = pulp.LpVariable("c", lowBound=0.0, upBound=1.0)
        w_vars = [pulp.LpVariable(f"w_{i}", lowBound=1e-8) for i in range(self.n)]

        prob += c
        prob += pulp.lpSum(w_vars) == 1.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                prob += w_vars[i] - w_vars[j] * upper_matrix[i, j] <= 1.0 * (1 - c)
                prob += -w_vars[i] + w_vars[j] * lower_matrix[i, j] <= 1.0 * (1 - c)

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30, gapRel=1e-10)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] == "Optimal":
            c_value = c.varValue
            weights = np.array([w.varValue for w in w_vars], dtype=float)
            weights = weights / np.sum(weights)
            return weights, float(c_value)
        else:
            raise RuntimeError(f"FPP не удался. Статус: {pulp.LpStatus[prob.status]}")

    def compute_weight_bounds(self, lower_matrix: np.ndarray, upper_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление мин и макс возможных весов."""
        w_min = np.zeros(self.n, dtype=float)
        w_max = np.zeros(self.n, dtype=float)

        for k in range(self.n):
            # Минимальный вес
            prob_min = pulp.LpProblem(f"Min_w_{k}", pulp.LpMinimize)
            w_vars_min = [pulp.LpVariable(f"w_{i}_min", lowBound=1e-8) for i in range(self.n)]
            prob_min += w_vars_min[k]
            prob_min += pulp.lpSum(w_vars_min) == 1.0

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    prob_min += w_vars_min[i] <= upper_matrix[i, j] * w_vars_min[j]
                    prob_min += w_vars_min[i] >= lower_matrix[i, j] * w_vars_min[j]

            prob_min.solve(pulp.PULP_CBC_CMD(msg=False))
            w_min[k] = float(pulp.value(w_vars_min[k]))

            # Максимальный вес
            prob_max = pulp.LpProblem(f"Max_w_{k}", pulp.LpMaximize)
            w_vars_max = [pulp.LpVariable(f"w_{i}_max", lowBound=1e-8) for i in range(self.n)]
            prob_max += w_vars_max[k]
            prob_max += pulp.lpSum(w_vars_max) == 1.0

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    prob_max += w_vars_max[i] <= upper_matrix[i, j] * w_vars_max[j]
                    prob_max += w_vars_max[i] >= lower_matrix[i, j] * w_vars_max[j]

            prob_max.solve(pulp.PULP_CBC_CMD(msg=False))
            w_max[k] = float(pulp.value(w_vars_max[k]))

        return w_min, w_max

    # ==================== ПОЛНЫЙ АНАЛИЗ ====================

    def run_complete_analysis(self, t: float = 3.0, s: float = 0.5) -> Dict:
        """
        Запуск полного анализа (применяется центр → СКО → интервалы → веса).
        Все вычисления унифицированы!
        """
        # 1. Вычисляем матрицу центра
        print(f"\n1. Вычисление центра (метод: {self.center_method.upper()})...")
        print("-" * 50)
        
        center_matrix = self.compute_center_matrix()
        std_matrix = self.compute_geometric_std_matrix(center_matrix)

        print(f"\nМатрица центра ({self.center_method.upper()}):")
        for i in range(self.n):
            row = " [ "
            for j in range(self.n):
                row += f"{center_matrix[i, j]:8.4f} "
            row += "]"
            print(row)

        print(f"\nМатрица геометрического стандартного отклонения:")
        for i in range(self.n):
            row = " [ "
            for j in range(self.n):
                row += f"{std_matrix[i, j]:8.4f} "
            row += "]"
            print(row)

        # 2. Оптимизируем лямбда
        print(f"\n2. Оптимизация параметра лямбда")
        print("-" * 50)

        self.lambda_opt = self.solve_model_1(t, s)
        lower_matrix, upper_matrix = self.compute_group_interval_matrix(self.lambda_opt)

        U = self.compute_indeterminacy_index(lower_matrix, upper_matrix)
        GSI = self.compute_group_satisfaction_index(lower_matrix, upper_matrix)

        print(f"Оптимальное лямбда: {self.lambda_opt:.4f}")
        print(f"Индекс неопределённости U: {U:.4f}")
        print(f"Индекс групповой удовлетворённости GSI: {GSI:.4f}")

        # 3. Интервальная матрица
        print(f"\n3. ИНТЕРВАЛЬНАЯ ГРУППОВАЯ МАТРИЦА")
        print("-" * 50)

        for i in range(self.n):
            row = " [ "
            for j in range(self.n):
                if i == j:
                    row += "1.0000 "
                else:
                    row += f"[{lower_matrix[i, j]:6.4f},{upper_matrix[i, j]:6.4f}] "
            row += "]"
            print(row)

        # 4. Вычисляем веса
        print(f"\n4. ВЫЧИСЛЕНИЕ ВЕСОВ КРИТЕРИЕВ")
        print("-" * 50)

        try:
            weights, c_value = self.solve_model_2_fpp(lower_matrix, upper_matrix)

            if abs(c_value - 1.0) < 1e-6:
                w_min, w_max = self.compute_weight_bounds(lower_matrix, upper_matrix)
                weights = (w_min + w_max) / 2
                weights = weights / np.sum(weights)

            self.group_weights = weights
            self.group_interval_matrix = (lower_matrix, upper_matrix)

            print("\nФинальные групповые веса критериев:")
            for i, (name, w) in enumerate(zip(self.criteria_names, weights)):
                print(f" {i+1:2d}. {name:20} : {w:.6f} ({w*100:.1f}%)")

            print(f"\n РАНЖИРОВАНИЕ КРИТЕРИЕВ ПО ВАЖНОСТИ")

            sorted_indices = np.argsort(weights)[::-1]
            for rank, idx in enumerate(sorted_indices):
                print(f" Ранг {rank+1:2d}: {self.criteria_names[idx]:20} (вес = {weights[idx]:.6f})")

            return {
                'center_method': self.center_method,
                'lambda_opt': self.lambda_opt,
                'U': U,
                'GSI': GSI,
                'weights': weights.tolist(),
                'weights_dict': {name: float(w) for name, w in zip(self.criteria_names, weights)},
                'ranking': [int(idx) for idx in sorted_indices.tolist()],
                'ranking_names': [self.criteria_names[idx] for idx in sorted_indices],
                'interval_matrix': {
                    'lower': lower_matrix.tolist(),
                    'upper': upper_matrix.tolist()
                },
                'c_value': float(c_value),
                'criteria_names': self.criteria_names,
                'dm_ids': self.dm_ids,
                'alpha_weights': self.alpha_weights
            }

        except Exception as e:
            print(f"Ошибка в FPP: {e}")
            raise

    # ==================== TROPICAL HELPERS ====================
    # (для метода tropical)

    @staticmethod
    def _max_times_matmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        Z = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                Z[i, j] = np.max(X[i, :] * Y[:, j])
        return Z

    @staticmethod
    def _kleene_star(A: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        S = np.eye(n, dtype=float)
        P = np.eye(n, dtype=float)
        for _ in range(1, n):
            P = GSDIJ_AHP._max_times_matmul(P, A)
            S = np.maximum(S, P)
        return S

    @staticmethod
    def _collinear(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
        r = a / b
        return (np.max(r) / np.min(r)) <= (1.0 + tol)

    @staticmethod
    def _independent_columns(S: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        n = S.shape[0]
        cols = []
        for j in range(n):
            c = S[:, j].copy()
            if not cols:
                cols.append(c)
                continue
            if any(GSDIJ_AHP._collinear(c, prev, tol=tol) for prev in cols):
                continue
            cols.append(c)
        return np.stack(cols, axis=1)

    @staticmethod
    def _best_diff_vector_from_star(S: np.ndarray) -> np.ndarray:
        Q = GSDIJ_AHP._independent_columns(S)
        n, r = Q.shape

        ratios = np.array([(np.max(Q[:, j]) / np.min(Q[:, j])) for j in range(r)], dtype=float)
        k = int(np.argmax(ratios))

        l = int(np.argmin(Q[:, k]))

        q_lk = float(Q[l, k])
        row_k = Q[l, :] / q_lk
        M = np.eye(r, dtype=float)
        M[k, :] = np.maximum(M[k, :], row_k)

        ones = np.ones(r, dtype=float)
        t = np.zeros(r, dtype=float)
        for i in range(r):
            t[i] = np.max(M[i, :] * ones)

        x = np.zeros(n, dtype=float)
        for i in range(n):
            x[i] = np.max(Q[i, :] * t)

        return x

    @staticmethod
    def _worst_diff_vector_from_matrix(B: np.ndarray, mu: float, S: np.ndarray) -> np.ndarray:
        n = B.shape[0]
        delta = float(np.max(S))
        J = np.ones((n, n), dtype=float)
        M = np.maximum(B / mu, J / delta)
        T = GSDIJ_AHP._kleene_star(M)
        x = np.max(T, axis=1)
        return x

    @staticmethod
    def trop_matmul_max_times(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.max(A[:, :, None] * B[None, :, :], axis=1)

    @staticmethod
    def trop_trace_max_times(A: np.ndarray) -> float:
        return float(np.max(np.diag(A)))

    @staticmethod
    def trop_spectral_radius_trace_formula(M: np.ndarray) -> float:
        n = M.shape[0]
        Ak = M.copy()
        lam = -np.inf

        for k in range(1, n + 1):
            tr = GSDIJ_AHP.trop_trace_max_times(Ak)
            lam = max(lam, tr ** (1.0 / k))
            if k < n:
                Ak = GSDIJ_AHP.trop_matmul_max_times(Ak, M)

        return float(lam)

    def save_interval_and_ranking(self, filepath: str, interval_matrix: Optional[tuple] = None) -> Dict[str, Any]:
        """Сохранение результатов в JSON."""
        if interval_matrix is None:
            interval_matrix = self.group_interval_matrix

        if interval_matrix is None:
            raise RuntimeError("Интервальная матрица не вычислена")

        if self.group_weights is None:
            raise RuntimeError("Веса не вычислены")

        lower, upper = interval_matrix
        w = np.asarray(self.group_weights, dtype=float)

        order = np.argsort(w)[::-1]
        ranking = []
        for rank, idx in enumerate(order, start=1):
            ranking.append({
                "rank": int(rank),
                "criterion_index": int(idx),
                "criterion_name": self.criteria_names[idx] if idx < len(self.criteria_names) else str(idx),
                "weight": float(w[idx]),
            })

        payload = {
            "center_method": self.center_method,
            "criteria_names": self.criteria_names,
            "weights": w.tolist(),
            "ranking": ranking,
            "interval_matrix": {
                "lower": np.asarray(lower).tolist(),
                "upper": np.asarray(upper).tolist(),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload


# # ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

# if __name__ == "__main__":
#     import sys

#     # Тестируем все три метода
#     methods = ["wgmm", "rowsum", "tropical"]
    
#     for method in methods:
#         print(f"\n{'='*70}")
#         print(f"ТЕСТ: center_method = '{method}'")
#         print(f"{'='*70}")
        
#         try:
#             gsd = GSDIJ_AHP(debug_mode=True, center_method=method)
#             gsd.load_from_json("data/pcm_output.json")
            
#             results = gsd.run_complete_analysis(t=5, s=0.3)
#             gsd.save_interval_and_ranking(f"results_unified_{method}.json")
            
#             print(f"\n✓ Метод '{method}' выполнен успешно!")
#             print(f"Оптимальный λ: {results['lambda_opt']:.4f}")
#             print(f"Индекс неопределённости: {results['U']:.4f}")
#             print(f"Индекс удовлетворённости: {results['GSI']:.4f}")
            
#         except FileNotFoundError as e:
#             print(f"⚠ Файл не найден: {e}")
#             print("Убедитесь, что файл 'data/pcm_output.json' существует.")
#         except Exception as e:
#             print(f"✗ Ошибка при методе '{method}': {e}")
#             import traceback
#             traceback.print_exc()
