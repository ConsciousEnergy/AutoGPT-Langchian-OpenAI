import os
from apikey import pineconeapi, pineconeenv, pineconeindex
import pinecone
from colorama import Fore, Style

os.environ['PINECONE_API_KEY'] = pineconeapi
os.environ['PINECONE_ENVIRONMENT'] = pineconeenv
os.environ['PINECONE_INDEX'] = pineconeindex

from autogpt.logs import logger
from autogpt.memory.base import MemoryProviderSingleton
from autogpt.llm_utils import create_embedding_with_ada


class PineconeMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pineconeapi, environment=pineconeenv)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "LENR_GPT"
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with
        #  memory.
        self.vec_num = 0

        try:
            pinecone.whoami()
        except Exception as e:
            logger.typewriter_log(
                "FAILED TO CONNECT TO PINECONE",
                Fore.RED,
                Style.BRIGHT + str(e) + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured Pinecone properly for use."
                + f"You can check out {Fore.CYAN + Style.BRIGHT}"
                "https://github.com/Torantulino/Auto-GPT#-pinecone-api-key-setup"
                f"{Style.RESET_ALL} to ensure you've set up everything correctly."
            )
            exit(1)

        if table_name not in pinecone.list_indexes():
            pinecone.create_index(
                table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = create_embedding_with_ada(data)
        # no metadata here. We may wish to change that long term.
        self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = create_embedding_with_ada(data)
        results = self.index.query(
            query_embedding, top_k=num_relevant, include_metadata=True
        )
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item["metadata"]["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()