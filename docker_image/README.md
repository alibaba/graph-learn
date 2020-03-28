# Build your own graph-learn docker image

To build this image on x86_64:

For CPU version:
```
docker build -f Dockerfile.cpu -t graphlearn/graphlearn:v0.1-cpu ./
```

For GPU version:
```
docker build -f Dockerfile.gpu -t graphlearn/graphlearn:v0.1-gpu ./
```

If you need to use the docker image on your cluster, you need to commit the docker image to a docker repo using TAG and PUSH. Please refer to the commands below.

```
docker tag graphlearn/graphlearn:v0.1-gpu <path-to-the-image-of-your-docker-repo>

docker push <path-to-the-image-of-your-docker-repo>
```

Note that, accessing the docker image from docker repo is essential for launching distributed graph-learn jobs with tf-operator.

[Home](../README.md)
