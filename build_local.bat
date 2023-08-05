

# =========================================
#     build and push docker image
# =========================================

cp ../osais_ai_base/main_common.py ./_temp/
cp ../osais_ai_base/main_fastapi.py ./_temp/
cp ../osais_ai_base/main_flask.py ./_temp/
cp ../osais_ai_base/osais_debug.py ./_temp/
cp ../osais_ai_base/osais.json ./_temp/

cp ./Dockerfile_sadtalker ./Dockerfile

docker build -t yeepeekoo/public:ai_sadtalker .  
