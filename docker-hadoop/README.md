# Big-Data-G4: Docker application with Hadoop and Spark installed
This is a Docker container with Hadoop, Spark and Jupyter installed.

The Jupyter Server Web UI can be accessed at http://localhost:8888.

## Quick start
To run the container, you need to have Docker installed on your machine (of course). Then, in your CLI of choice, navigate to this folder and run:
```
docker-compose up -d
```
to run the container in detached mode.

## What services are available?
The Hadoop service is run with images provided by the [big-data-europe](https://github.com/big-data-europe) project, the repo can be found at [docker-hadoop](https://github.com/big-data-europe/docker-hadoop). You may access the WebHDFS service through http://localhost:9870.\
However, for clients connecting to the WebHDFS service, you need to connect this client **WITHIN** the appropriate container and through the address `http://namenode:9000`. Check [`crawling.ipynb`](../review_crawl/crawling.ipynb) for an example.

The Jupyter service with Pyspark installed is run with the [pyspark-notebook](https://hub.docker.com/r/jupyter/pyspark-notebook) image provided by [jupyter](https://jupyter.org/).

## Using Jupyter Notebook
To use the Jupyter Server in your notebook session, when selecting a kernel, select `Existing Jupyter Server...` and specify the connection as http://localhost:8888. When entered, you will be prompted to connect to an insecure network, this is because the server is set up to be **passwordless** and require no token authentication.

This is fine as you are connecting to your locally hosted server. However, in case you want to share your server to others, specify a password or include token authentication by changing line 51 in [`docker-compose.yml`](./docker-compose.yml).

Note that when running, the notebook is running within the container's file system. Any data on your host machine is shared through the [`data-mount`](./data-mount/) folder.

## References
The `docker-compose.yml` and `hadoop.env` file is provided by [hadoop-spark](https://github.com/OneCricketeer/docker-stacks/tree/master/hadoop-spark) with minor adjustments to suit this project.