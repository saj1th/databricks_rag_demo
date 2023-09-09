# Databricks notebook source
# MAGIC %run ./includes/_common

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch-preview
# MAGIC %pip install pprintpp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

vs_catalog = "vs_catalog"
embedding_model_endpoint = "e5-small-v2"
vs_index_fullname = "vs_catalog.demo.delta_docs_chunked"

# COMMAND ----------

vsc.list_indexes(vs_catalog)

# COMMAND ----------

vsc.create_index(
  source_table_name="dbx_sa_datasets.delta_docs.delta_docs_chunked",
  dest_index_name=vs_index_fullname,
  primary_key="idx",
  index_column="text",
  embedding_model_endpoint_name=embedding_model_endpoint
)

# COMMAND ----------

def wait_for_index_to_be_ready(index_name):
  for i in range(180):
    if vsc.get_index(index_name)['index_status']['state'] == 'NOT_READY':
      if i % 20 == 0: print(vsc.get_index(index_name)['index_status']['message'])
      time.sleep(10)
    else:
      return vsc.get_index(index_name)
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name)}")

# COMMAND ----------

import pprintpp
pprintpp.pprint(vsc.get_index(vs_index_fullname))

# COMMAND ----------


