# Docker container with Hadoop and Spark installed
This is a Docker container with Hadoop, Spark and Jupyter installed.

The Jupyter Server Web UI can be opened at http://localhost:8888.

## How do I run the container?
To run the container, you need to have Docker installed on your machine (of course). Then, in your CLI of choice, navigate to this folder and run:
```
docker-compose up -d
```
to run the container in detached mode.

## References
The `docker-compose.yml` and `hadoop.env` file is provided by [hadoop-spark](https://github.com/OneCricketeer/docker-stacks/tree/master/hadoop-spark) with minor adjustments to suit this project.