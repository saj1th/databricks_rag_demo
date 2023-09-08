# Databricks notebook source
# MAGIC %run ./includes/_common

# COMMAND ----------

movie_df = spark.read.table("movies_raw")

# COMMAND ----------

display(movie_df)

# COMMAND ----------

import mlflow

encoder_uri = "models:/llm-content-discovery-encoder/Production"
encoder = mlflow.pyfunc.spark_udf(spark, encoder_uri, result_type="array<float>")

# COMMAND ----------

# DBTITLE 1,Apply the encoder to the data
encoded_df = movie_df.withColumn("text_embedding", encoder("corpus"))

# COMMAND ----------

display(encoded_df)

# COMMAND ----------

from pyspark.sql import functions as F

encoded_df.filter(F.col("original_language") == "en").filter(~F.col("genres").contains("Documentary")).write.mode("overwrite").saveAsTable("movies_embeddings")

# COMMAND ----------


