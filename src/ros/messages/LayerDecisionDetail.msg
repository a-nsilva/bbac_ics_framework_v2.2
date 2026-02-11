# BBAC Framework - Layer Decision Detail
# Individual layer decision within hybrid approach
# Score + decision + confidence + latency + explanation
# Uso: agregado em AccessDecision.layer_decisions[]
# Resultado final consolidado

string layer_name          # Layer name: 'statistical', 'sequence', 'policy', 'fusion', 'rbac_enforcement'
string decision            # Layer decision: 'grant' or 'deny'
float64 confidence         # Layer confidence (0.0 to 1.0)
float64 latency_ms         # Layer latency in milliseconds
float64 score             # Score contínuo [0,1] (já está em dataclass)
#string[] explanation_keys  # Explanation keys (e.g., ['aggregate_score', 'threshold'])
#string[] explanation_values # Explanation values as JSON strings
string explanation_json   # JSON serializado de Dict[str, Any]
