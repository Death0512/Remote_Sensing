from dataclasses import dataclass
from typing import Mapping, Optional, Union
from src.domain.enums import MotionClass, TargetClass


@dataclass(frozen=True)
class MotionModelConfig:
    motion_class: MotionClass
    process_noise_pos: float
    process_noise_vel: float
    min_speed_mps: float
    max_speed_mps: float
    max_dt_seconds: float
    chi2_gate_threshold: float
    default_measurement_noise_std: float
    max_acceleration_mps2: float


STATIONARY_MODEL = MotionModelConfig(
    motion_class=MotionClass.STATIONARY,
    process_noise_pos=0.0001,
    process_noise_vel=0.0,
    min_speed_mps=0.0,
    max_speed_mps=0.5,
    max_dt_seconds=86400.0,
    chi2_gate_threshold=5.991,
    default_measurement_noise_std=5.0,
    max_acceleration_mps2=0.05,
)

VESSEL_MODEL = MotionModelConfig(
    motion_class=MotionClass.VESSEL,
    process_noise_pos=0.1,
    process_noise_vel=0.5,
    min_speed_mps=0.0,
    max_speed_mps=20.577,
    max_dt_seconds=7200.0,
    chi2_gate_threshold=9.210,
    default_measurement_noise_std=10.0,
    max_acceleration_mps2=1.0,
)

AIRCRAFT_MODEL = MotionModelConfig(
    motion_class=MotionClass.AIRCRAFT,
    process_noise_pos=1.0,
    process_noise_vel=10.0,
    min_speed_mps=50.0,
    max_speed_mps=310.0,
    max_dt_seconds=1800.0,
    chi2_gate_threshold=13.816,
    default_measurement_noise_std=25.0,
    max_acceleration_mps2=30.0,
)

UNKNOWN_MODEL = MotionModelConfig(
    motion_class=MotionClass.UNKNOWN,
    process_noise_pos=0.5,
    process_noise_vel=2.0,
    min_speed_mps=0.0,
    max_speed_mps=50.0,
    max_dt_seconds=3600.0,
    chi2_gate_threshold=9.210,
    default_measurement_noise_std=15.0,
    max_acceleration_mps2=5.0,
)

MOTION_MODELS: dict[MotionClass, MotionModelConfig] = {
    MotionClass.STATIONARY: STATIONARY_MODEL,
    MotionClass.VESSEL: VESSEL_MODEL,
    MotionClass.AIRCRAFT: AIRCRAFT_MODEL,
    MotionClass.UNKNOWN: UNKNOWN_MODEL,
}


def target_class_to_motion_class(
    target_class: Optional[Union[TargetClass, MotionClass, str]] = None,
) -> MotionClass:
    if target_class is None:
        return MotionClass.UNKNOWN
    if isinstance(target_class, MotionClass):
        return target_class
    val = target_class.value if hasattr(target_class, "value") else str(target_class)
    val_upper = val.upper()
    if val_upper == "VESSEL":
        return MotionClass.VESSEL
    if val_upper in ("AIRCRAFT", "AIRCRAFT_INFLIGHT"):
        return MotionClass.AIRCRAFT
    if val_upper in (
        "AIRCRAFT_PARKED",
        "STATIONARY",
        "INFRASTRUCTURE_CHANGE",
        "NEW_STRUCTURE",
        "UNKNOWN_STRUCTURE",
    ):
        return MotionClass.STATIONARY
    try:
        return MotionClass(val_upper)
    except ValueError:
        return MotionClass.UNKNOWN


def get_motion_model(
    motion_or_target: Optional[Union[MotionClass, TargetClass, str]] = None,
    overrides: Optional[Mapping[MotionClass, MotionModelConfig]] = None,
) -> MotionModelConfig:
    m_class = target_class_to_motion_class(motion_or_target)
    if overrides and m_class in overrides:
        return overrides[m_class]
    return MOTION_MODELS[m_class]
