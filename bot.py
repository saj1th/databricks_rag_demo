# Databricks notebook source


# COMMAND ----------

import pinecone
import os

from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Databricks
from langchain.chains import RetrievalQA


class Bot:
    def __init__(self):
        pinecone.init(
            api_key=config["pinecone"]["api_key"], environment=config["pinecone"]["env"]
        )
        pinecone_index_name = config["pinecone"]["index_name"]
        pinecone_index = pinecone.Index(pinecone_index_name)
        os.environ["DATABRICKS_TOKEN"] = config["dbx"]["api_key"]

        model_kwargs = {}
        encode_kwargs = {"normalize_embeddings": False}
        hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        vectordb = Pinecone(pinecone_index, hf.embed_query, "title", namespace="tmdb")

        template = """You are a helpful assistant built by Databricks, you are good at
                        helping to answer a question based on the context provided, the
                        context is a document.

                        If the context does not provide enough relevant information to
                        determine the answer, just say I don't know. If the context is
                        irrelevant to the question, just say I don't know. If you did not
                        find a good answer from the context, just say I don't know. If the
                        query doesn't form a complete question, just say I don't know.

                        If there is a good answer from the context, try to summarize the
                        context to answer the question.
                Context: {context}
                Question: {question}
                Helpful Answer:"""

        qa_chain_prompt = PromptTemplate.from_template(template)

        llm = Databricks(
            host=config["dbx"]["host"],
            endpoint_name="llamav2-13b-chat",
            model_kwargs={"temperature": 0.5, "max_tokens": 500},
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

    def query(self, question):
      response = self.qa_chain({"query": question})
      return response['result']

# COMMAND ----------

# bot = Bot()
# bot.query("Could you recommend me a good martial arts movie")

# COMMAND ----------


