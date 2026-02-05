# BBAC Framework - Layer Decision Detail
# Individual layer decision within hybrid approach

string layer_name          # Layer name: 'statistical', 'sequence', 'policy', 'fusion', 'rbac_enforcement'
string decision            # Layer decision: 'grant' or 'deny'
float64 confidence         # Layer confidence (0.0 to 1.0)
float64 latency_ms         # Layer latency in milliseconds
string[] explanation_keys  # Explanation keys (e.g., ['aggregate_score', 'threshold'])
string[] explanation_values # Explanation values as JSON strings