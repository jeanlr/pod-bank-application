#!/bin/bash
set -e

# Ajusta permissão para o usuário jovyan nas pastas montadas (ignora erros se pastas não existirem)
chown -R jovyan:jovyan /data /home/jovyan/artifacts /home/jovyan/work /home/jovyan/code || true

# Executa o comando passado como usuário jovyan
exec su jovyan -c "$*"
