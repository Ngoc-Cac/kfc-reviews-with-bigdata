# Big-Data-G4: Sentiment Anlysis
This is the core of the project where data preprocessing, model training and data anlytics are done. The preprocessing and data analytics are mostly done with Spark. For ML models, we utilise Spark's built in MLlib as well as PyTorch for Deep Learning models.\
For information on how to install PyTorch as well as how to set up GPU resources for the Docker application to use, see []().

This section includes:
- [Preprocessing](#preprocessing)
- [EDA](#eda)
- [Model Training](#model-training)

## Preprocessing
As mentioned, the whole project will process data using Spark, more specifically Python API for Spark called [PySpark](https://spark.apache.org/docs/latest/api/python/index.html). The preprocessing steps include:
1. Bucketizing rating scores (1-5) into sentiments (positive, neutral, negative) for model training. Ratings above 3 stars are consider positive, below 3 stars are considered negative and neutral otherwise.
2. Vietnamese word tokenization using [PyVi](https://github.com/trungtv/pyvi)'s tokenizer.
3. Replacing abbreviated words with its full form.

After preprocessing, the data is saved back into the HDFS for data anlytics. Furthermore, to create convenience for model training, the preprocessed data is also splitted into train and test set using stratification and saved into HDFS.

## EDA


## Model Training


---
---
## Installing PyTorch and Sharing GPU resources to Docker Container