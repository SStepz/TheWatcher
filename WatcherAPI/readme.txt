# ML model deployment example

Complete code (including a trained model) to deploy and inference a machine learning model (built on the iris dataset) using Docker and FastAPI.

1. With terminal navigate to the root of this repository

---

2. Build docker image

    #. Normal
        $ docker build -t image_name .

    #. For Multi-Architecture
        $ docker buildx build --push -t sstepz/watcher-rec:v1 --platform linux/amd64,linux/arm64 .

---

3. Run container

    $ docker run --name container_name -p 80:80 -e MONGO_USERNAME="" -e MONGO_PASSWORD="" sstepz/watcher-rec:v1

---

4. Output will contain

    INFO: Uvicorn running on http://0.0.0.0:80

    Use this url in chrome to see the model frontend;
    use http://0.0.0.0:80/docs for testing the model in the web interface.

---

5. Query model

    #. Via web interface (chrome):
        http://0.0.0.0:80/docs -> test model

    #. Via python client:
        client.py

    #. Via curl request:

        $ curl -X POST "http://0.0.0.0:80/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
