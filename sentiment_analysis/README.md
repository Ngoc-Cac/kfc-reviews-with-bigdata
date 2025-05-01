# Big-Data-G4: Sentiment Anlysis
This is the core of the project where data preprocessing, model training and data anlytics are done.

## The Pipeline
The project aims to build a small-scale Big Data system that stores information about Google Places (address, price range, reviews). The data are stored using Apache Hadoop's HDFS and are processed with Apache Spark through PySpark API.

To deploy the HDFS, we opt for prebuilt Docker images as these are lightweight and easy to deploy. Furthermore, we also deploy a container within the Docker application to run PySpark and other Python scripts. Further information can be found in [`docker-hadoop/README.md`](../docker-hadoop/README.md). The Docker application is hosted locally on our machine.

When the Docker application is set up, we are then free to proceed as needed. The first task was to crawl reviews data. We did this by using [`selenium`](https://pypi.org/project/selenium/) to simulate the website for user-input interactions. Then when data are crawled from a website, we save the result to the hosted HDFS with [`hdfs`](https://pypi.org/project/hdfs/) (A Python API for WebHDFS). The details on how this works is provided at [`review_crawl/README.md`](../review_crawl/README.md).

Finally, after crawling, we proceeded to preprocess the data and fine-tune some models as well as use the preprocessed data for EDA.