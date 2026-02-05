# BBAC Framework - Emergency Alert Message

# Header with timestamp
std_msgs/Header header

# Emergency information
string emergency_type      # Type: 'fire', 'power_outage', 'equipment_failure', 'safety_breach', etc.
string severity            # Severity: 'low', 'medium', 'high', 'critical'
uint8 priority             # Priority level (0-100, higher = more urgent)

# State
bool active                # Whether emergency is currently active
string status              # Status: 'triggered', 'active', 'clearing', 'cleared'

# Location
string zone                # Affected zone
string[] affected_resources  # List of affected resources

# Actions
string[] required_actions    # Actions to be taken (e.g., "stop_all_robots")
string[] notifications       # Who to notify (e.g., ["supervisor", "safety_officer"])

# Context
string description         # Human-readable description
string[] context_data      # Additional context information