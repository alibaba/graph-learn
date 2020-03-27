# Build your own graph-learn docker image

To build this image on x86_64:
```
docker build -f Dockerfile -t graphlearn/graphlearn:v0.1 ./
```

If you need to use the docker image on your cluster, you need to commit the docker image to a docker repo using TAG and PUSH. Please refer to the commands below.

```
docker tag graphlearn/graphlearn:v0.1 <path-to-the-image-of-your-docker-repo>

docker push <path-to-the-image-of-your-docker-repo>
```

Note that, accessing the docker image from docker repo is essential for launching distributed graph-learn jobs with tf-operator.

[Home](../README.md)
