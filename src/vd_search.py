import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import openai
import weaviate.classes as wvc

openai_apikey = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_apikey

# Load environment variables
load_dotenv()
weaviate_url = os.getenv('WEAVIATE_URL')
weaviate_apikey = os.getenv('WEAVIATE_API_KEY')

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Initialize Weaviate client
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(api_key=os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Check connection
if not client.is_ready():
    print("Weaviate is not ready. Check your credentials or connection.")
    exit()


# create the schema for the vector store
collection_name = 'gmail_chunk' 

if client.collections.exists(collection_name):
    client.collections.delete(collection_name) 

client_chunk = client.collections.create(
    name=collection_name,
    properties=[
        
        wvc.config.Property(
            name="subject", 
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="sender", 
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="receiver", 
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="timestamp", 
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="snippet", 
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="chunk_index", 
            data_type=wvc.config.DataType.INT  # Index to track chunk order
        ),
        wvc.config.Property(
            name="chunk", 
            data_type=wvc.config.DataType.TEXT  # The actual chunk of text
        )
    ],
    vectorizer_config = wvc.config.Configure.Vectorizer.text2vec_openai(),  # OpenAI text vectorization
    generative_config = wvc.config.Configure.Generative.openai()           # Generative config for OpenAI
)


def email_chunks(mail):
    """
        Function to create chunks
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n", ".", " ", ""]
    )
    
    for email in mail:
        sender = email.get("sender", "Unknown")
        receiver = email.get("receiver", "Unknown")
        timestamp = email.get("timestamp", "1970-01-01T00:00:00")
        subject = email.get("subject", "No Subject")
        snippet = email.get("snippet", "")

        # Split only the snippet into chunks
        email_snippet_chunks = text_splitter.split_text(snippet)

        # Store each chunk separately while keeping metadata intact
        for idx, chunk in enumerate(email_snippet_chunks):
            chunk_object = {
                "chunk": chunk,
                "chunk_index": idx,
                "subject": subject,
                "sender": sender,
                "receiver": receiver,
                "timestamp": timestamp,
                "snippet": snippet  # Full snippet for reference (can remove if not needed)
            }
            all_chunks.append(chunk_object)
    return all_chunks






# Load email data
with open('D:/MailMind/src/emails.json', 'r') as f:
    email_data = json.load(f)

# Handle multiple emails
if isinstance(email_data, list):  # JSON is an array of emails
    emails = email_data
else:  # JSON is a single email object
    emails = [email_data]



chunked_text = email_chunks(emails)




chunks_list = list()
j=0
for i, chunk in enumerate(chunked_text):
    # print("chunk:", chunk)
    embedding_vector = embeddings.embed_documents([chunk["chunk"]])[0]
    j+=1
    # embedding_vector = embeddings.embed_documents([chunk])[0]
    data_properties = {
        "chunk": chunk['chunk'],
        "subject": chunk['subject'],
        "sender": chunk['sender'],
        "receiver": chunk['receiver'],
        "timestamp": chunk['timestamp'],
        "snippet": chunk['snippet'],
        "chunk_index": i
    }
    # data_object = wvc.data.DataObject(properties=data_properties , vector=embedding_vector)
    
    client_chunk.data.insert(properties= data_properties , vector= embedding_vector)




response = client_chunk.aggregate.over_all(total_count=True)
print(f"Total stored objects: {response.total_count}")

response = client_chunk.generate.near_text(
    query='emails American Express <americanexpress@member.americanexpress.com>',
    limit=2,
    grouped_task="give a clear message, Summarize this message"
)

print(" ********************** Request fulfilled using weaviate vector store **************")
print(response.generated)