# Databricks notebook source
# MAGIC %run ./includes/_common

# COMMAND ----------

from pyspark.sql.functions import concat
from pyspark.sql import functions as F

df = (
    spark.read.format("csv")
    .option("delimiter", ",")
    .option("inferSchema", "true")
    .option("header", "true")
    .load(config["all"]["staging_loc"])
    .drop("poster_path", "backdrop_path", "recommendations")
)
txt_lst = ["title", "genres", "overview", "credits", "keywords"]

df = df.withColumn("corpus", concat_ws('-', *txt_lst))

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from transformers import AutoTokenizer
max_length = 128
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


@pandas_udf("long")
def num_tokens_mpt7b(s: pd.Series) -> pd.Series:
  return s.apply(lambda str: len(tokenizer.encode(str)))

# COMMAND ----------

df = df.withColumn("num_tokens_mpt7b", num_tokens_mpt7b("corpus"))

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("movies_raw")

# COMMAND ----------


