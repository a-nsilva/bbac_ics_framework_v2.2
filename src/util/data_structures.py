#!/usr/bin/env python3
"""
BBAC ICS Framework - Data Structures

Centralized definitions for the entire system:
- Enums: All enumeration types (ROS-compatible with explicit values)
- Types: Type aliases and custom types
- Dataclasses: Structured data containers
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# TYPE ALIASES
# ============================================================================

AgentID = str
RequestID = str
ResourceID = str
SessionID = str
Timestamp = float
Confidence = float  # 0.0 to 1.0
LatencyMS = float   # milliseconds


# ============================================================================
# ENUMS - TYPE DEFINITIONS (ROS-COMPATIBLE)
# ============================================================================

class AgentType(Enum):
    """Type of agent (robot vs human)."""
    ROBOT = "robot"
    HUMAN = "human"
    
    @classmethod
    def from_string(cls, value: str) -> 'AgentType':
        """Parse from dataset string."""
        value_lower = value.lower().strip()
        if value_lower in ("robot", "r"):
            return cls.ROBOT
        elif value_lower in ("human", "h", "operator"):
            return cls.HUMAN
        raise ValueError(f"Unknown AgentType: {value}")


class AgentRole(Enum):
    """Role within the system."""
    # Human roles
    SUPERVISOR = "supervisor"
    OPERATOR = "operator"
    TECHNICIAN = "technician"
    
    # Robot roles
    ASSEMBLY_ROBOT = "assembly_robot"
    CAMERA_ROBOT = "camera_robot"
    TRANSPORT_ROBOT = "transport_robot"
    INSPECTION_ROBOT = "inspection_robot"
    SAFETY_ROBOT = "safety_robot"
    
    # Fallback
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str, agent_type: AgentType = None) -> 'AgentRole':
        """Parse from dataset (robot_type or human_role field)."""
        if not value or value == "":
            return cls.UNKNOWN
        
        value_lower = value.lower().strip()
        
        # Direct mapping
        mapping = {
            "assembly_robot": cls.ASSEMBLY_ROBOT,
            "camera_robot": cls.CAMERA_ROBOT,
            "transport_robot": cls.TRANSPORT_ROBOT,
            "inspection_robot": cls.INSPECTION_ROBOT,
            "safety_robot": cls.SAFETY_ROBOT,
            "supervisor": cls.SUPERVISOR,
            "operator": cls.OPERATOR,
            "technician": cls.TECHNICIAN,
        }
        
        if value_lower in mapping:
            return mapping[value_lower]
        
        # Try to infer from agent_id pattern (robot_assembly_16 → assembly_robot)
        if "assembly" in value_lower:
            return cls.ASSEMBLY_ROBOT
        elif "camera" in value_lower:
            return cls.CAMERA_ROBOT
        elif "transport" in value_lower:
            return cls.TRANSPORT_ROBOT
        elif "inspection" in value_lower:
            return cls.INSPECTION_ROBOT
        elif "safety" in value_lower:
            return cls.SAFETY_ROBOT
        
        return cls.UNKNOWN


class ActionType(Enum):
    """Types of actions agents can perform."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    OVERRIDE = "override"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_OVERRIDE = "emergency_override"
    TRANSPORT = "transport"
    MAINTENANCE = "maintenance"
    MONITOR = "monitor"
    CALIBRATION = "calibration"
    DIAGNOSTIC = "diagnostic"
    
    @classmethod
    def from_string(cls, value: str) -> 'ActionType':
        """Parse from dataset string."""
        value_lower = value.lower().strip()
        
        mapping = {
            "read": cls.READ,
            "write": cls.WRITE,
            "execute": cls.EXECUTE,
            "delete": cls.DELETE,
            "override": cls.OVERRIDE,
            "emergency_stop": cls.EMERGENCY_STOP,
            "emergency_override": cls.EMERGENCY_OVERRIDE,
            "transport": cls.TRANSPORT,
            "maintenance": cls.MAINTENANCE,
            "monitor": cls.MONITOR,
            "calibration": cls.CALIBRATION,
            "diagnostic": cls.DIAGNOSTIC,
        }
        
        if value_lower in mapping:
            return mapping[value_lower]
        
        raise ValueError(f"Unknown ActionType: {value}")


class AuthStatus(Enum):
    """Authentication result status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    LOCKED = "locked"
    
    @classmethod
    def from_string(cls, value: str) -> 'AuthStatus':
        """Parse from dataset string."""
        value_lower = value.lower().strip()
        
        if value_lower in ("success", "granted", "ok", "true"):
            return cls.SUCCESS
        elif value_lower in ("failed", "denied", "false"):
            return cls.FAILED
        elif value_lower in ("timeout",):
            return cls.TIMEOUT
        elif value_lower in ("locked", "blocked"):
            return cls.LOCKED
        
        return cls.FAILED


class AgentBehavior(Enum):
    """Agent behavior patterns."""
    PREDICTABLE = "predictable"
    NORMAL = "normal"
    VARIABLE = "variable"
    ADVERSARIAL = "adversarial"
    SCHEDULED = "scheduled"
    MAINTENANCE_DRIVEN = "maintenance_driven"


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_RESOURCE = "unauthorized_resource"
    HIGH_FREQUENCY = "high_frequency"
    WRONG_SEQUENCE = "wrong_sequence"
    BRUTE_FORCE = "brute_force"
    TIME_VIOLATION = "time_violation"


class DecisionType(Enum):
    """Access control decision types."""
    GRANT = "grant"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    UNCERTAIN = "uncertain"


class EmergencyType(Enum):
    """Types of emergency events."""
    FIRE = "fire"
    POWER_OUTAGE = "power_outage"
    EQUIPMENT_FAILURE = "equipment_failure"
    SAFETY_BREACH = "safety_breach"
    GAS_LEAK = "gas_leak"
    NETWORK_ATTACK = "network_attack"
    HUMAN_INJURY = "human_injury"


class FusionStrategy(Enum):
    """Decision fusion strategies for combining 3 layers."""
    RULE_PRIORITY = "rule_priority"
    HIGH_CONFIDENCE_DENIAL = "high_confidence_denial"
    WEIGHTED_VOTING = "weighted_voting"


class ResourceType(Enum):
    """Types of resources - Generic categories + dynamic string support."""
    # Hardware/Physical
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    CAMERA = "camera"
    ROBOT_ARM = "robot_arm"
    CONVEYOR = "conveyor"
    
    # Stations
    ASSEMBLY_STATION = "assembly_station"
    INSPECTION_AREA = "inspection_area"
    WAREHOUSE = "warehouse"
    MAINTENANCE_BAY = "maintenance_bay"
    
    # Storage/Data
    DATABASE = "database"
    FILE_STORAGE = "file_storage"
    IMAGE_STORAGE = "image_storage"
    LOG_STORAGE = "log_storage"
    
    # Control/Admin
    ADMIN_PANEL = "admin_panel"
    PRODUCTION_SCHEDULE = "production_schedule"
    SAFETY_SYSTEM = "safety_system"
    EMERGENCY_CONTROLS = "emergency_controls"
    NETWORK_GATEWAY = "network_gateway"
    ACCESS_CONTROL = "access_control"
    POWER_CONTROL = "power_control"
    
    # Generic fallback
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> 'ResourceType':
        """Parse from dataset resource_type field."""
        if not value:
            return cls.OTHER
        
        value_lower = value.lower().strip()
        
        # Direct mapping
        for member in cls:
            if member.value == value_lower:
                return member
        
        # Fuzzy matching for common patterns
        if "actuator" in value_lower:
            return cls.ACTUATOR
        elif "sensor" in value_lower:
            return cls.SENSOR
        elif "camera" in value_lower:
            return cls.CAMERA
        elif "database" in value_lower or "db" in value_lower:
            return cls.DATABASE
        elif "assembly" in value_lower:
            return cls.ASSEMBLY_STATION
        elif "warehouse" in value_lower or "storage" in value_lower:
            return cls.WAREHOUSE
        elif "maintenance" in value_lower:
            return cls.MAINTENANCE_BAY
        
        return cls.OTHER


class RequestFrequency(Enum):
    """Request frequency levels with time ranges in seconds."""
    VERY_HIGH = (1, 2)      # 1-2 seconds
    HIGH = (3, 5)           # 3-5 seconds
    MEDIUM = (5, 8)         # 5-8 seconds
    LOW = (10, 15)          # 10-15 seconds
    VERY_LOW = (20, 30)     # 20-30 seconds
    
    def get_range(self) -> Tuple[int, int]:
        """Get min/max seconds for this frequency level."""
        return self.value
    
    @classmethod
    def from_string(cls, frequency: str) -> 'RequestFrequency':
        """Get enum from string."""
        try:
            return cls[frequency.upper()]
        except KeyError:
            return cls.MEDIUM


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for any agent (robot or human)."""
    id: str
    type: AgentType
    role: AgentRole
    behavior: AgentBehavior
    allowed_actions: List[ActionType]
    allowed_resources: List[ResourceType]
    
    def __post_init__(self):
        """Validate agent configuration."""
        if not self.id:
            raise ValueError("Agent ID cannot be empty")
        if not self.allowed_actions:
            raise ValueError(f"Agent {self.id} must have at least one allowed action")


@dataclass
class AccessRequest:
    """
    Access request format used across all modules.
    - 17: log_id, timestamp, session_id, user_id, agent_type, 
        robot_type, human_role, action, resource, resource_type,
        human_present, emergency_flag, location, previous_action,
        auth_status, attempt_count, policy_id
    """
    # Core identifiers (log_id, timestamp, user_id)
    request_id: str
    timestamp: float
    agent_id: str
    
    # Agent information (agent_type, robot_type/human_role → agent_role)
    agent_type: AgentType
    agent_role: AgentRole
    
    # Request details
    action: ActionType
    resource: str  # Dynamic string (e.g., "database_inventory", "actuator_arm_02")
    resource_type: ResourceType  # Categorical type
    
    # Context
    location: str  # e.g., "assembly_line", "maintenance_bay"
    human_present: bool = False
    emergency: bool = False #emergency_flag
    
    # Session tracking
    session_id: Optional[str] = None
    previous_action: Optional[ActionType] = None
    
    # Authentication
    auth_status: AuthStatus = AuthStatus.SUCCESS
    attempt_count: int = 0
    
    # Policy tracking
    policy_id: Optional[str] = None
    
    # Legacy/optional fields
    zone: Optional[str] = None  # Alias for location
    priority: float = 5.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp <= 0:
            self.timestamp = time.time()
        if not 1.0 <= self.priority <= 10.0:
            raise ValueError(f"Priority must be between 1.0 and 10.0, got {self.priority}")
        
        # Sync zone with location if not set
        if self.zone is None and self.location:
            self.zone = self.location


@dataclass
class AccessDecision:
    """Access decision format used across all modules."""
    request_id: str
    timestamp: float
    decision: DecisionType
    confidence: float
    latency_ms: float
    reason: str
    layer_decisions: Dict[str, Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate decision parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.latency_ms < 0:
            raise ValueError(f"Latency cannot be negative, got {self.latency_ms}")


@dataclass
class LayerWeights:
    """Weight configuration for layer fusion."""
    rule: float = 0.4
    behavioral: float = 0.3
    ml: float = 0.3
    
    def __post_init__(self):
        """Validate weights."""
        for name, value in [("rule", self.rule), ("behavioral", self.behavioral), ("ml", self.ml)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Weight '{name}' must be in [0,1], got {value}")
        total = sum([self.rule, self.behavioral, self.ml])
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class LayerDecision:
    """Decision from a single layer."""
    decision: str  # 'grant' or 'deny'
    confidence: float
    latency_ms: float
    layer_name: str = ""
    explanation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate decision."""
        if self.decision not in {"grant", "deny"}:
            raise ValueError(f"Invalid decision: {self.decision}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        if self.latency_ms < 0:
            raise ValueError("Latency cannot be negative")


@dataclass
class HybridDecision:
    """Final fused decision."""
    request_id: str
    decision: str
    confidence: float
    fusion_strategy: FusionStrategy
    layer_results: Dict[str, LayerDecision]
    total_latency_ms: float
    explanation: Dict[str, Any]
    
    def __post_init__(self):
        """Validate final decision."""
        if self.decision not in {"grant", "deny"}:
            raise ValueError("Invalid final decision")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Invalid confidence")
        if self.total_latency_ms < 0:
            raise ValueError("Latency cannot be negative")


@dataclass
class BBACConfig:
    """BBAC Engine configuration."""
    enable_rule: bool = True
    enable_behavioral: bool = True
    enable_ml: bool = True
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_VOTING
    weights: LayerWeights = field(default_factory=LayerWeights)
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.enable_rule:
            raise ValueError("Rule layer must always be enabled")
    
    def enabled_layers(self) -> List[str]:
        """Get list of enabled layers."""
        layers = []
        if self.enable_rule:
            layers.append("rule")
        if self.enable_behavioral:
            layers.append("behavioral")
        if self.enable_ml:
            layers.append("ml")
        return layers


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    tp: int
    tn: int
    fp: int
    fn: int
    fpr: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    precision_curve: List[float] = field(default_factory=list)
    recall_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1,
            "roc_auc": self.roc_auc,
            "average_precision": self.avg_precision,
            "confusion_matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn,
            },
        }


@dataclass
class LatencyMetrics:
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    values: List[float]

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class PerformanceMetrics:
    latency: LatencyMetrics
    throughput: float
    total_requests: int
    total_time: float

    def to_dict(self) -> Dict:
        return {
            "latency_stats": self.latency.to_dict(),
            "throughput": self.throughput,
            "total_requests": self.total_requests,
            "processing_time": self.total_time,
        }


@dataclass
class StatisticalTest:
    p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    significant: bool

    def to_dict(self) -> Dict:
        return {
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "ci_95": [self.ci_low, self.ci_high],
            "significant": self.significant,
        }
    

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    scenario: str
    max_requests: int
    enable_behavioral: bool = True
    enable_ml: bool = True
    use_real_dataset: bool = True
    dataset_split: str = 'test'  # 'train', 'validation', 'test'
    save_results: bool = True
    output_dir: str


@dataclass 
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    metrics: Dict
    ground_truth: List[str]
    predictions: List[str]
    latencies: List[float]
    execution_time: float
    success: bool


__all__ = [
    # Type aliases
    'AgentID',
    'RequestID',
    'ResourceID',
    'SessionID',
    'Timestamp',
    'Confidence',
    'LatencyMS',

    # Enums
    'AgentType',
    'AgentRole',
    'ActionType',
    'AuthStatus',
    'AgentBehavior',
    'AnomalyType',
    'DecisionType',
    'EmergencyType',
    'FusionStrategy',
    'ResourceType',
    'RequestFrequency',

    # Dataclasses
    'AgentConfig', 
    'AccessRequest', 
    'AccessDecision', 
    'LayerWeights',
    'LayerDecision', 
    'HybridDecision', 
    'BBACConfig',
    'ClassificationMetrics',
    'LatencyMetrics',
    'PerformanceMetrics',
    'StatisticalTest',
    'ExperimentConfig',
    'ExperimentResult',
]