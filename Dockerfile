FROM python:3.9-slim
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]
