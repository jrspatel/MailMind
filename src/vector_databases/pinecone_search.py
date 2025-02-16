from pinecone.grpc import PineconeGRPC as Pinecone 
from pinecone import ServerlessSpec 
import time 
import os 
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv 
load_dotenv()


client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key= api_key) 
# Define a sample dataset where each item has a unique ID, text, and category
# data = [
#     {
#         "id": "rec1",
#         "text": "🍏 Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut. 🦠",
#         "category": "digestive system"
#     },
#     {
#         "id": "rec2",
#         "text": "🌍 Apples originated in Central Asia and have been cultivated for thousands of years, with over 7,500 varieties available today. 🍎",
#         "category": "cultivation"
#     },
#     {
#         "id": "rec3",
#         "text": "💪 Rich in vitamin C 🍊 and other antioxidants, apples contribute to immune health 🛡️ and may reduce the risk of chronic diseases. 🍏",
#         "category": "immune system"
#     },
#     {
#         "id": "rec4",
#         "text": "📉 The high fiber content in apples 🍏 can also help regulate blood sugar levels, making them a favorable snack for people with diabetes. 🩸",
#         "category": "endocrine system"
#     }
# ]

with open('D:/MailMind/src/emails.json', 'r') as f:
    email_data = json.load(f)

# for data in email_data:
#     print(data) 
#     qwhdiu

# converting the text into embeddings

# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')


# embeddings = pc.inference.embed(
#     model= 'text-embedding-3-small',
#     inputs= [d['snippet'] for d in email_data] , 
#     parameters= {
#         'input_type' : 'passage',
#         'truncate' : 'END'
#     }

# ) 

response = client.embeddings.create(
    model = 'text-embedding-ada-002',
    input = [d['snippet'] for d in email_data]
)


embeddings = [record.embedding for record in response.data]
print(len(embeddings)) 
print(len(embeddings[0]))

# creating an serveless index 
index_name = 'e-mail-index1' 

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name= index_name, 
        dimension= len(embeddings[0]), 
        metric= 'dotproduct', # depending on the type of embeddings usage
        spec= ServerlessSpec(
            cloud= 'aws',
            region= 'us-east-1'
        )
    ) 

# creating index takes time, adding a timer  
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1) 

# adding the records into VD - index - namespace
index = pc.Index(index_name) # called targetting the index - in productn through host-variable 

records = [] 
for (d,e) in (zip(email_data, embeddings)): 
    records.append({
        'id' : d['id'], 
        'values' : e,
        'metadata': {
            'source_text': d['snippet'],
            'sender': d['sender'],
            'timestamp' : d['timestamp']
        }
    }) 

index.upsert(
    vectors= records,
    namespace= 'ex-name'
)

time.sleep(10)  # Wait for the upserted vectors to be indexed

print(index.describe_index_stats())





query = 'Amex on 2025-01-09' 

# query_embed = pc.inference.embed(
#     model= 'multilingual-e5-large',
#     inputs= [query],
#     parameters= {
#         'input_type': 'query'
#     }
# ) 

xq = client.embeddings.create(input=query, model='text-embedding-ada-002').data[0].embedding

filter_date_str = "2025-01-09"
filter_date_numeric = int(datetime.strptime(filter_date_str, "%Y-%m-%d").timestamp())
# Search the index for the three most similar vectors
results = index.query(
    namespace="ex-name",
    vector=xq,
    top_k=3,
    include_values=False,
    include_metadata=True,
    filter={
    "sender": "amex",
    "timestamp": {"$gte": filter_date_numeric}
}
)

print(results)