from enum import Enum


class Capability(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    HEURISTIC = "HEURISTIC"
    EXTERNAL_MODEL = "EXTERNAL_MODEL"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


class Zone(str, Enum):
    OUTSIDE_AOI = "OUTSIDE_AOI"
    ZONE_1 = "ZONE_1"
    ZONE_2 = "ZONE_2"
    ZONE_3 = "ZONE_3"

    ZONE_1_CRITICAL = "ZONE_1"
    ZONE_2_HIGH = "ZONE_2"
    ZONE_3_MEDIUM = "ZONE_3"


class QualityTier(str, Enum):
    DATA_OK = "DATA_OK"
    DATA_PARTIAL = "DATA_PARTIAL"
    CLOUD_CONTAMINATED = "CLOUD_CONTAMINATED"
    GEOREGISTRATION_UNCERTAIN = "GEOREGISTRATION_UNCERTAIN"
    RADIOMETRIC_UNCERTAIN = "RADIOMETRIC_UNCERTAIN"
    OUTSIDE_VALID_FOOTPRINT = "OUTSIDE_VALID_FOOTPRINT"
    STALE = "STALE"
    CORRUPTED_METADATA = "CORRUPTED_METADATA"

    HIGH = "DATA_OK"
    PARTIAL = "DATA_PARTIAL"


class AisCorrelationState(str, Enum):
    MATCHED = "MATCHED"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    NO_MATCH = "NO_MATCH"
    AIS_UNAVAILABLE = "AIS_UNAVAILABLE"
    AIS_STALE = "AIS_STALE"
    CORRELATION_UNCERTAIN = "CORRELATION_UNCERTAIN"

    AIS_MATCHED = "MATCHED"
    AIS_NO_MATCH_WITHIN_VERIFIED_COVERAGE = "NO_MATCH"


class TargetClass(str, Enum):
    VESSEL = "VESSEL"
    AIRCRAFT = "AIRCRAFT"
    INFRASTRUCTURE_CHANGE = "INFRASTRUCTURE_CHANGE"
    UNIDENTIFIED = "UNIDENTIFIED"
    NEW_STRUCTURE = "NEW_STRUCTURE"
    UNKNOWN_STRUCTURE = "UNKNOWN_STRUCTURE"
    AIRCRAFT_PARKED = "AIRCRAFT_PARKED"
    AIRCRAFT_INFLIGHT = "AIRCRAFT_INFLIGHT"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other or self.value == other.upper() or self.name == other.upper()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


class HypothesisStatement(str, Enum):
    DARK_VESSEL_CANDIDATE = "DARK_VESSEL_CANDIDATE"
    DARK_VESSEL_TRANSITING = "DARK_VESSEL_TRANSITING"
    UNVERIFIED_MARITIME_TARGET = "UNVERIFIED_MARITIME_TARGET"
    UNVERIFIED_SURFACE_TARGET = "UNVERIFIED_SURFACE_TARGET"
    ANOMALOUS_TERRAIN_CHANGE = "ANOMALOUS_TERRAIN_CHANGE"
    INFRASTRUCTURE_EXPANSION_DETECTED = "INFRASTRUCTURE_EXPANSION_DETECTED"
    UNIDENTIFIED_SURFACE_OBJECT = "UNIDENTIFIED_SURFACE_OBJECT"
    SURFACE_VESSEL_LOITERING = "SURFACE_VESSEL_LOITERING"
    EXPERIMENTAL_INFLIGHT_TARGET = "EXPERIMENTAL_INFLIGHT_TARGET"
    UNIDENTIFIED_AIR_INCURSION = "UNIDENTIFIED_AIR_INCURSION"


class EvidenceKind(str, Enum):
    OBJECT_CONFIDENCE = "OBJECT_CONFIDENCE"
    CHANGE_STRENGTH = "CHANGE_STRENGTH"
    TEMPORAL_PERSISTENCE = "TEMPORAL_PERSISTENCE"
    SPATIAL_CONTEXT = "SPATIAL_CONTEXT"
    AIS_CORRELATION = "AIS_CORRELATION"
    SENSOR_QUALITY = "SENSOR_QUALITY"
    KINEMATICS = "KINEMATICS"


class AlertPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    STRATEGIC = "STRATEGIC"
    INFO = "INFO"
    INFORMATIONAL = "INFORMATIONAL"


class ReviewAction(str, Enum):
    CONFIRM = "CONFIRM"
    DISMISS = "DISMISS"
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"


class ReviewState(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    UNCERTAIN = "UNCERTAIN"
    PENDING_REVIEW = "PENDING_REVIEW"
    FLAGGED_UNCERTAIN = "FLAGGED_UNCERTAIN"


class CalibrationState(str, Enum):
    NOT_CALIBRATED = "NOT_CALIBRATED"
    PLATT = "PLATT"
    ISOTONIC = "ISOTONIC"
    RELIABILITY_CURVE = "RELIABILITY_CURVE"


class SensorType(str, Enum):
    S1_SAR = "S1_SAR"
    S2_MSI = "S2_MSI"
    VHR_OPTICAL = "VHR_OPTICAL"


class RadiometricState(str, Enum):
    RAW = "RAW"
    CALIBRATED_SIGMA0 = "CALIBRATED_SIGMA0"
    CALIBRATED_GAMMA0 = "CALIBRATED_GAMMA0"
    TOA_REFLECTANCE = "TOA_REFLECTANCE"
    SURFACE_REFLECTANCE = "SURFACE_REFLECTANCE"


class GeometricState(str, Enum):
    GEOCODED = "GEOCODED"
    TERRAIN_CORRECTED = "TERRAIN_CORRECTED"
    UNCORRECTED = "UNCORRECTED"


class TrackLifecycle(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    TERMINATED = "TERMINATED"


class ScoreMethod(str, Enum):
    RULE_ENGINE = "RULE_ENGINE"
    BAYESIAN = "BAYESIAN"
    CALIBRATED_CLASSIFIER = "CALIBRATED_CLASSIFIER"
    NONE = "NONE"


class GeometryStatus(str, Enum):
    GEOCODED = "GEOCODED"
    MAP_ONLY = "MAP_ONLY"
    PIXEL_ONLY = "PIXEL_ONLY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other or self.value == other.upper() or self.name == other.upper()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


class MotionClass(str, Enum):
    STATIONARY = "STATIONARY"
    VESSEL = "VESSEL"
    AIRCRAFT = "AIRCRAFT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other or self.value == other.upper() or self.name == other.upper()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


CalibrationStatus = CalibrationState
