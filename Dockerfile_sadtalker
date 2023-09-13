##
##      To build the AI_SADTALKER docker image
##

# base stuff
FROM yeepeekoo/public:ai_sadtalker_

## keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai

## todo : copy all
COPY ./ai/checkpoints ./ai/checkpoints
COPY ./ai/gfpgan ./ai/gfpgan
COPY ./ai/scripts ./ai/scripts
COPY ./ai/src ./ai/src
COPY ./ai/runai.py ./ai/runai.py

# push again the base files
COPY ./_temp/static/* ./static
COPY ./_temp/templates/* ./templates
COPY ./_temp/osais.json .
COPY ./_temp/main_fastapi.py .
COPY ./_temp/main_flask.py .
COPY ./_temp/main_common.py .
COPY ./_temp/osais_debug.py .

# copy OSAIS mapping into AI
COPY ./sadtalker.json .
COPY ./_input/warmup.jpg ./_input/warmup.jpg
COPY ./_input/warmup.wav ./_input/warmup.wav

# overload config with those default settings
ENV ENGINE=sadtalker

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5309"]
