FROM ultralytics/ultralytics:latest-python as build

ENV YOLO_MODEL_NAME yolov8n

WORKDIR /tmp

COPY ./requirements.txt /tmp/requirements.txt

RUN  mkdir -p /output/app \
    && apt-get update && apt-get install -y pkg-config default-libmysqlclient-dev build-essential\
    && pip3 install --target /output/app -r ./requirements.txt \
    && rm -rf /output/app/*-*info

# outcommented since we use our own model
#RUN curl -LJO https://github.com/ultralytics/assets/releases/download/v0.0.0/${YOLO_MODEL_NAME}.pt \
#    && yolo export model=${YOLO_MODEL_NAME}.pt format=onnx \
#    && mv ${YOLO_MODEL_NAME}.onnx /output/app/model.onnx

COPY ./src/ /output/app

# finalize
FROM ultralytics/ultralytics:latest-python

COPY --from=build /output/app /app

EXPOSE 5000

WORKDIR /app
ENTRYPOINT [ "python3", "-u", "main.py" ]

HEALTHCHECK --interval=1m --timeout=10s \
  CMD python3 /app/healthcheck.py