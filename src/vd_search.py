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

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

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

# Function to create chunks
def email_chunks(mail):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Adjust based on your needs
        chunk_overlap=50,  # Ensure context continuity
        separators=["\n", ".", " ", ""]
    )
    
    for email in mail:
        # Combine metadata and content
        combined_data = (
            f"Sender: {email['sender']}\n"
            f"Receiver: {email['receiver']}\n"
            f"Timestamp: {email['timestamp']}\n"
            f"Subject: {email['subject']}\n"
            f"Snippet: {email['snippet']}"
        )
        
        # Split text into chunks
        email_chunks = text_splitter.split_text(combined_data)
        
        # Add chunks directly to the list
        for chunk in email_chunks:
            all_chunks.append(chunk)
    
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
            data_type=wvc.config.DataType.DATE
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


chunks_list = list()
for i, chunk in enumerate(chunked_text):
     
    data_properties = {
        "chunk": chunk,
        "chunk_index": i
    }
    data_object = wvc.data.DataObject(properties=data_properties)
    chunks_list.append(data_object)

client_chunk.data.insert_many(chunks_list)

response = client_chunk.aggregate.over_all(total_count=True)
print(response.total_count)

response = client_chunk.generate.near_text(
    query='Summarize the emails I received on [yesterday\'s date - 1/24/2025], if there are any. Include the timestamp of the email.',
    limit=2,
    grouped_task="if there are no emails, give a clear message, Summarize this message"
)

print(" ********************** Request fulfilled using weaviate vector store **************")
print(response.generated)