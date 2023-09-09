# Databricks notebook source
# MAGIC %run ./includes/_common

# COMMAND ----------

df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "text")
    .option("wholeText", "True")
    .load(config["all"]["staging_loc"])
)

(
    df.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", config["all"]["base_loc"] + "/checkpoint/bronze")
    .toTable("raw_data")
)

# COMMAND ----------

# df = spark.readStream.table("raw_data")
df = spark.read.table("raw_data")

# COMMAND ----------

# MAGIC %sql SELECT count (*) from raw_data;

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter
from pyspark.sql import functions as F
from pyspark.sql import types as T

@udf('array<string>')
def get_chunks(text):
  # instantiate tokenization utilities
  text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
  # split text into chunks
  return text_splitter.split_text(text)


# split text into chunks
df_chunked = (
  df
    .withColumn('chunks', get_chunks('value')) # divide text into chunks
    .drop('value')
    .withColumn('chunk', F.expr("explode(chunks)"))
    .drop('chunks')
    .withColumnRenamed('chunk','text')
    .withColumn("idx", F.monotonically_increasing_id())
  )

# 

# COMMAND ----------

display(df_chunked)

# COMMAND ----------

df_chunked.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable("delta_docs_chunked")


# COMMAND ----------

# MAGIC %sql SELECT count(*) FROM delta_docs_chunked;

# COMMAND ----------

# MAGIC %sql SELECT * FROM delta_docs_chunked;

# COMMAND ----------


