# Docker container with Hadoop and Spark installed
This is a Docker container with Hadoop, Spark and Jupyter installed.

The Jupyter Server Web UI can be opened at http://localhost:8888.

## How do I run the container?
To run the container, you need to have Docker installed on your machine (of course). Then, in your CLI of choice, navigate to this folder and run:
```
docker-compose up -d
```
to run the container in detached mode.

## Using Jupyter Notebook
To use the Jupyter Server in your notebook session, when selecting a kernel, select a Jupyter Server and specify the connection as `http://localhost:8888`. When entered, you will be prompted to connect an insecure network, this is because the conenction is passwordless and require no token authentication.

This is fine as you are connecting to your locally hosted server. However, in case you need to create a server for others to connect to, specify a password by changing line 51 in [`docker-compose.yml`](./docker-compose.yml).

Note that when running, the notebook is running within the container's file system. Any data in your host machine is synced through the [`data-mount`](./data-mount/) folder.

## References
The `docker-compose.yml` and `hadoop.env` file is provided by [hadoop-spark](https://github.com/OneCricketeer/docker-stacks/tree/master/hadoop-spark) with minor adjustments to suit this project.