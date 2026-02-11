# BBAC_ICS - LayerOutput.msg
# scores intermediários
string layer_name          # Layer name: 'statistical', 'sequence', 'policy', 'fusion', 'rbac_enforcement'
float32 score
float32 confidence         # Layer confidence (0.0 to 1.0)
string[] metadata
