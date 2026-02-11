# BBAC_ICS - LayerOutput.msg
# scores intermediários
#score + confidence + metadata
#Uso: publicação em tópicos intermediários (/bbac/layer/statistical/output)
#Streaming em tempo real para monitoramento/debug

string layer_name          # Layer name: 'statistical', 'sequence', 'policy', 'fusion', 'rbac_enforcement'
float32 score
float32 confidence         # Layer confidence (0.0 to 1.0)
string[] metadata
