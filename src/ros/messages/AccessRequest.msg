# BBAC Framework - Access Request Message
# Aligned with AccessRequest dataclass (17 fields from dataset)

# Header with timestamp
std_msgs/Header header

# Core identifiers
string request_id           # Unique request identifier (log_id from dataset)
string agent_id            # Agent identifier (user_id from dataset)

# Agent information
string agent_type          # Type: 'robot' or 'human'
string agent_role          # Role: 'assembly_robot', 'supervisor', etc. (robot_type/human_role)

# Request details
string action              # Action: 'read', 'write', 'execute', 'delete', etc.
string resource            # Resource identifier (e.g., 'database_inventory', 'actuator_arm_02')
string resource_type       # Resource type: 'database', 'actuator', 'sensor', etc.

# Context
string location            # Physical location: 'assembly_line', 'maintenance_bay', etc.
bool human_present         # Whether human is present
bool emergency             # Emergency flag (emergency_flag from dataset)

# Session tracking
string session_id          # Session identifier (optional)
string previous_action     # Previous action in sequence (for Markov analysis, optional)

# Authentication (new fields)
string auth_status         # Auth status: 'success', 'failed', 'timeout', 'locked'
uint8 attempt_count        # Number of authentication attempts

# Policy tracking (new field)
string policy_id           # Applied policy identifier (optional)

# Optional fields
string zone                # Alias for location (backward compatibility)
float64 priority           # Request priority (1.0 to 10.0), default: 5.0
#string[] context_data      # Additional context as key-value pairs
string context_json       # JSON serializado de Dict[str, Any]

string ground_truth       # Ground truth label ('allow', 'deny', 'mfa', 'review') - apenas datasets
