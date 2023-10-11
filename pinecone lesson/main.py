# pinecone.io

import pinecone
from langchain.document_loaders import PyPDFLoader
import pinecone 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PDFMinerLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

pinecone.init(api_key="63ff77f5-3860-4c19-ac55-68a43d1a145f", environment="gcp-starter")

# once created -> comment
# pinecone.create_index('hubi-idx', dimension=1536, metric="cosine")
# print(pinecone.describe_index("hubi-idx"))


#pinecone.create_index("quickstart", dimension=1536, metric="cosin")
#pinecone.describe_index("quickstart") 

print(pinecone.list_indexes())
index = pinecone.Index("quickstart")

loader = PyPDFLoader("https://cdn.revolut.com/terms_and_conditions/pdf/personal_terms_e59ea0d9_1.2.2_1695110904_en.pdf")
pages = loader.load_and_split()
print(pages)


embed = OpenAIEmbeddings(
    openai_api_key="OPENAI_KEY"
)

Pinecone.from_documents(pages, embed, index_name="hubi-idx")

'''
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(openai_api_key=my_openai_api_key)

# ładowanie do indeksu:
from langchain.vectorstores import Pinecone

# odczytywanie indeksu:
query = "Co się stanie jak nie pójdę do szkoły?"

index = Pinecone.from_existing_index("hubi-idx", embeddings_model)
# result = index.similarity_search_with_relevance_scores(query)

# for document in result:
# 	print(document[0].page_content)


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import StdOutCallbackHandler

qa = retrievalQA.from_chain_type(llm=OpenAI(openai_api_key=my_openai_api_key),
				 chain_type="stuff",
				 retriever=index.as_retriever(),
				 callbacks = [StdOutCallbackHandler()])

result = qa.run(query)
print(result)
'''
