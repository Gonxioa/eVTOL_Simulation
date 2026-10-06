"""Conventional bounded control allocation for the hexarotor.

The allocator deliberately uses the *nominal* (healthy-aircraft) allocation
matrix.  Actuator effectiveness belongs to the simulated plant, not to this
controller.  This separation is important for fault-diagnosis data: feeding
the ground-truth fault factor into the controller would both hide the fault
signature and leak the answer that a diagnostic model is meant to infer.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class AllocationResult:
    """Result of one nominal control-allocation solve.

    ``success`` reports numerical solver success.  ``wrench_feasible`` is a
    separate flag: an infeasible requested wrench is a valid allocator result
    as long as the returned commands remain finite and inside their bounds.
    Residuals use the convention ``desired - allocated``.
    """

    desired_wrench: np.ndarray
    commanded_thrusts: np.ndarray
    allocated_wrench: np.ndarray
    residual_wrench: np.ndarray
    lower_bound_active: np.ndarray
    upper_bound_active: np.ndarray
    success: bool
    wrench_feasible: bool
    cost: float
    weighted_residual_norm: float
    bound_active_count: int
    used_unconstrained_solution: bool


def validate_effectiveness(effectiveness, num_rotors):
    """Return a validated actuator-effectiveness vector in ``[0, 1]``."""
    values = np.asarray(effectiveness, dtype=float)
    if values.shape != (int(num_rotors),):
        raise ValueError(
            "effectiveness must have shape "
            f"({int(num_rotors)},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("effectiveness contains NaN or Inf")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("effectiveness values must lie in [0, 1]")
    return values.copy()


def scheduled_effectiveness(
    t,
    num_rotors,
    fault_time=None,
    fault_motor_indices=(),
    fault_factor=1.0,
):
    """Build the simulation plant's effectiveness vector for one time step."""
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    factor = float(fault_factor)
    if not np.isfinite(factor) or factor < 0.0 or factor > 1.0:
        raise ValueError("fault_factor must be finite and lie in [0, 1]")

    count = int(num_rotors)
    indices = tuple(int(index) for index in fault_motor_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("fault_motor_indices must not contain duplicates")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(
            f"fault_motor_indices must be between 0 and {count - 1}"
        )

    values = np.ones(count, dtype=float)
    if fault_time is None or not indices or factor >= 1.0:
        return values

    fault_time = float(fault_time)
    if not np.isfinite(fault_time):
        raise ValueError("fault_time must be finite or None")
    if t >= fault_time:
        values[list(indices)] = factor
    return values


def predict_effective_response(allocation_matrix, commanded_thrusts, effectiveness):
    """Predict steady actuator thrusts and wrench after plant degradation.

    This is logging/validation math only; the nominal allocator does not use
    this prediction to alter its commands.
    """
    matrix = np.asarray(allocation_matrix, dtype=float)
    thrusts = np.asarray(commanded_thrusts, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 4:
        raise ValueError("allocation_matrix must have shape (4, num_rotors)")
    if thrusts.shape != (matrix.shape[1],):
        raise ValueError(
            f"commanded_thrusts must have shape ({matrix.shape[1]},)"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(thrusts).all():
        raise ValueError("allocation inputs contain NaN or Inf")
    eta = validate_effectiveness(effectiveness, matrix.shape[1])
    effective_thrusts = eta * thrusts
    return effective_thrusts, matrix @ effective_thrusts


class BoundedControlAllocator:
    """Allocate a four-axis wrench to bounded, nonnegative rotor thrusts.

    A feasible minimum-norm pseudoinverse solution is retained exactly.  If it
    violates a physical bound, a weighted bounded least-squares problem is
    solved.  The weights are normalized by the aircraft's per-axis wrench
    authority, so newtons and newton-metres are compared on physical scales.
    """

    def __init__(
        self,
        allocation_matrix,
        thrust_min,
        thrust_max,
        wrench_priority=(2.0, 1.0, 1.0, 0.2),
        wrench_scale=None,
        regularization=0.0,
        solver_tolerance=1e-10,
        feasibility_tolerance=1e-6,
        bound_tolerance=1e-8,
        max_iterations=200,
    ):
        matrix = np.asarray(allocation_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != 4:
            raise ValueError("allocation_matrix must have shape (4, num_rotors)")
        if matrix.shape[1] < 4 or not np.isfinite(matrix).all():
            raise ValueError(
                "allocation_matrix needs at least four finite rotor columns"
            )
        if np.linalg.matrix_rank(matrix) < 4:
            raise ValueError("nominal allocation_matrix must have row rank 4")

        self.allocation_matrix = matrix.copy()
        self.num_rotors = matrix.shape[1]
        self.thrust_min = self._as_bound_vector(thrust_min, "thrust_min")
        self.thrust_max = self._as_bound_vector(thrust_max, "thrust_max")
        if np.any(self.thrust_min < 0.0):
            raise ValueError("thrust_min must be nonnegative")
        if np.any(self.thrust_max <= self.thrust_min):
            raise ValueError("every thrust_max must be greater than thrust_min")

        priority = np.asarray(wrench_priority, dtype=float)
        if priority.shape != (4,) or not np.isfinite(priority).all():
            raise ValueError("wrench_priority must contain four finite values")
        if np.any(priority <= 0.0):
            raise ValueError("wrench_priority values must be positive")
        self.wrench_priority = priority.copy()

        if wrench_scale is None:
            max_abs_thrust = np.maximum(
                np.abs(self.thrust_min), np.abs(self.thrust_max)
            )
            scale = np.sum(
                np.abs(self.allocation_matrix) * max_abs_thrust[None, :],
                axis=1,
            )
        else:
            scale = np.asarray(wrench_scale, dtype=float)
        if scale.shape != (4,) or not np.isfinite(scale).all():
            raise ValueError("wrench_scale must contain four finite values")
        if np.any(scale <= 0.0):
            raise ValueError("wrench_scale values must be positive")
        self.wrench_scale = scale.copy()
        self.row_weights = self.wrench_priority / self.wrench_scale

        self.regularization = float(regularization)
        self.solver_tolerance = float(solver_tolerance)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.bound_tolerance = float(bound_tolerance)
        self.max_iterations = int(max_iterations)
        if not np.isfinite(self.regularization) or self.regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(self.solver_tolerance) or self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and positive")
        if (
            not np.isfinite(self.feasibility_tolerance)
            or self.feasibility_tolerance <= 0.0
        ):
            raise ValueError("feasibility_tolerance must be finite and positive")
        if not np.isfinite(self.bound_tolerance) or self.bound_tolerance <= 0.0:
            raise ValueError("bound_tolerance must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self._weighted_matrix = self.row_weights[:, None] * self.allocation_matrix
        self._pseudoinverse = np.linalg.pinv(self.allocation_matrix)
        thrust_range = self.thrust_max - self.thrust_min
        self._regularization_scale = np.maximum(thrust_range, 1e-12)

    def _as_bound_vector(self, value, name):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(self.num_rotors, float(array))
        if array.shape != (self.num_rotors,) or not np.isfinite(array).all():
            raise ValueError(
                f"{name} must be finite scalar or shape ({self.num_rotors},)"
            )
        return array.copy()

    def weighted_residual_norm(self, residual_wrench):
        residual = np.asarray(residual_wrench, dtype=float)
        if residual.shape != (4,) or not np.isfinite(residual).all():
            raise ValueError("residual_wrench must contain four finite values")
        return float(np.linalg.norm(self.row_weights * residual))

    def allocate(self, desired_wrench):
        desired = np.asarray(desired_wrench, dtype=float)
        if desired.shape != (4,):
            raise ValueError("desired_wrench must have shape (4,)")
        if not np.isfinite(desired).all():
            raise ValueError("desired_wrench contains NaN or Inf")

        candidate = self._pseudoinverse @ desired
        candidate_residual = desired - self.allocation_matrix @ candidate
        within_bounds = bool(
            np.all(candidate >= self.thrust_min - self.bound_tolerance)
            and np.all(candidate <= self.thrust_max + self.bound_tolerance)
        )
        exact_enough = (
            self.weighted_residual_norm(candidate_residual)
            <= self.solver_tolerance
        )

        used_unconstrained = within_bounds and exact_enough
        if used_unconstrained:
            thrusts = np.clip(candidate, self.thrust_min, self.thrust_max)
            success = True
        else:
            weighted_target = self.row_weights * desired
            solve_matrix = self._weighted_matrix
            solve_target = weighted_target
            if self.regularization > 0.0:
                reg_matrix = (
                    np.sqrt(self.regularization)
                    * np.diag(1.0 / self._regularization_scale)
                )
                solve_matrix = np.vstack((solve_matrix, reg_matrix))
                solve_target = np.concatenate(
                    (solve_target, np.zeros(self.num_rotors))
                )
            solution = lsq_linear(
                solve_matrix,
                solve_target,
                bounds=(self.thrust_min, self.thrust_max),
                method="trf",
                tol=self.solver_tolerance,
                lsmr_tol="auto",
                max_iter=self.max_iterations,
                verbose=0,
            )
            if not solution.success:
                raise RuntimeError(
                    "bounded control allocation failed: " + str(solution.message)
                )
            thrusts = np.asarray(solution.x, dtype=float)
            success = bool(solution.success)

        if not np.isfinite(thrusts).all():
            raise RuntimeError("bounded control allocation returned NaN or Inf")
        if (
            np.any(thrusts < self.thrust_min - self.bound_tolerance)
            or np.any(thrusts > self.thrust_max + self.bound_tolerance)
        ):
            raise RuntimeError("bounded control allocation violated thrust bounds")
        thrusts = np.clip(thrusts, self.thrust_min, self.thrust_max)

        allocated = self.allocation_matrix @ thrusts
        residual = desired - allocated
        weighted_residual = self.weighted_residual_norm(residual)
        normalized_residual = residual / self.wrench_scale
        wrench_feasible = bool(
            np.max(np.abs(normalized_residual)) <= self.feasibility_tolerance
        )

        active_tolerance = np.maximum(
            self.bound_tolerance,
            1e-7 * np.maximum(1.0, self.thrust_max - self.thrust_min),
        )
        lower_active = np.abs(thrusts - self.thrust_min) <= active_tolerance
        upper_active = np.abs(thrusts - self.thrust_max) <= active_tolerance
        bound_count = int(np.count_nonzero(lower_active | upper_active))

        regularization_residual = thrusts / self._regularization_scale
        cost = 0.5 * weighted_residual**2
        if self.regularization > 0.0:
            cost += 0.5 * self.regularization * float(
                regularization_residual @ regularization_residual
            )

        return AllocationResult(
            desired_wrench=desired.copy(),
            commanded_thrusts=thrusts.copy(),
            allocated_wrench=allocated.copy(),
            residual_wrench=residual.copy(),
            lower_bound_active=lower_active.copy(),
            upper_bound_active=upper_active.copy(),
            success=success,
            wrench_feasible=wrench_feasible,
            cost=float(cost),
            weighted_residual_norm=weighted_residual,
            bound_active_count=bound_count,
            used_unconstrained_solution=used_unconstrained,
        )
