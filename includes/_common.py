# Databricks notebook source
# DBTITLE 1,Dependencies
# %pip install git+https://github.com/huggingface/transformers@main
# %pip install -U "accelerate>=0.20.3"
%pip install pinecone-client

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

#Fetch secrets
secret_scope_key = config['all']['secret_scope_key']

config['dbx']['api_key'] = dbutils.secrets.get(secret_scope_key, config['dbx']['api_key'])


# COMMAND ----------

# DBTITLE 1,Set OS env vars
import os

os.environ["DATABRICKS_HOST"] = config['dbx']['host']
os.environ["DATABRICKS_TOKEN"] = config['dbx']['api_key']
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "True"


# COMMAND ----------

spark.catalog.setCurrentCatalog(config['dbx']['catalog'])
spark.catalog.setCurrentDatabase(config['dbx']['database'])

# COMMAND ----------

import huggingface_hub.utils
huggingface_hub.utils.disable_progress_bars()
