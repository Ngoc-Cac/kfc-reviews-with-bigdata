# Big-Data-G4: Sentiment Analysis on KFC reviews from Google Maps
This is a project on Sentiment Analysis using Big Data framework Apache Hadoop and Apache Spark. For ease of deployment, we opt for a multi-container Docker application to install the necessary service: Hadoop, Spark, Python and Jupyter.\
The project aims to develop a classification model on general comments about the chain KFC. In order to train the model, we crawled reviews from KFC locations within Hồ Chí Minh city on Google Maps.


#### Table of Contents
- [**Overview**](#overview)
    - [Deploying the Docker Application](#docker_deploy)
- [**References**](#ref)

## Overview
<a id="overview"></a>
The whole project can be best explained by the following figure:

![workflow](./resource/pngs/workflow.png)
*Figure: The project's overall workflow*

<a name="docker_deploy"></a>
<details>
<summary><b>Deploying the Docker Application</b></summary>

Taking advantage of pre-built Docker images, we opt for multi-container Docker application for quick deployment of the working environment.\
In the application, we deploy two images provided by [`big-data-europ/docker-hadoop`](https://github.com/big-data-europe/docker-hadoop). These images are responsible for the Hadoop HDFS service within the project.\
For processing with Spark, we use the official [`jupyter/pyspark-notebook`](https://hub.docker.com/r/jupyter/pyspark-notebook) image. The latest image has Python 3.11.6 installed with PySpark. The image also hosts a local JupyterLab session where one can easily connect to from outside of the Docker container.

For more information on the application, see [`docker-hadoop/README.md`](./docker-hadoop/README.md)
To see how one deploys, runs as well as connecting to the JupyterLab instance, visit [`how_to_setup.ipynb`](how_tos.ipynb).
</details>

<details>
<summary><b>Crawling Reviews from Google Maps</b></summary>


</details>
<details>
<summary><b>Storing Reviews on Hadoop HDFS and Processing data with PySpark</b></summary>


</details>
<details>
<summary><b>Model Training and Evaluation</b></summary>


</details>

<a name="ref"></a>

## References
### Our contributors:
<a href="https://github.com/Ngoc-Cac">
    <img src="https://avatars.githubusercontent.com/u/144905277?v=4" alt="drawing" width="60">
</a>
<a href="https://github.com/dothimykhanh">
    <img src="https://avatars.githubusercontent.com/u/120184309?v=4" alt="drawing" width="60">
</a>
<a href="https://github.com/NguyenTNTh">
    <img src="https://avatars.githubusercontent.com/u/203326835?v=4" alt="drawing" width="60">
</a>
<a href="https://github.com/hako1106">
    <img src="https://avatars.githubusercontent.com/u/117138002?v=4" alt="drawing" width="60">
</a>
<a href="https://github.com/phiyenng">
    <img src="https://avatars.githubusercontent.com/u/145342146?v=4" alt="drawing" width="60">
</a>