# BBAC Framework - Access Decision Message

# Header with timestamp
std_msgs/Header header

# Core decision
string request_id          # ID of the corresponding AccessRequest
string decision            # Decision: 'grant', 'deny', 'require_approval', 'uncertain'
float64 confidence         # Overall confidence (0.0 to 1.0)
float64 latency_ms         # Total decision latency in milliseconds
string reason              # Primary reason for the decision

# Layer decisions (detailed breakdown)
LayerDecisionDetail[] layer_decisions  # Array of layer-specific decisions

# Logging
bool logged                # Whether decision was logged
string log_id              # Log entry identifier (optional)
