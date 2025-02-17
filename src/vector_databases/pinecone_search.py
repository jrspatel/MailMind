from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import os
import json
from datetime import datetime
import redis
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


import redis

redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
print(f'redis client {redis_client}')

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

# Initialize Redis cache (local or remote)
# redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Load email data
with open('D:/MailMind/src/emails.json', 'r') as f:
    email_data = json.load(f)

# Generate embeddings for email snippets using OpenAI
response = client.embeddings.create(
    model='text-embedding-ada-002',
    input=[d.get('snippet', '') for d in email_data]
)

embeddings = [record.embedding for record in response.data]
print(len(embeddings))
print(len(embeddings[0]))

# Define Pinecone index
index_name = 'e-mail-index1'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(embeddings[0]),
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Wait for index creation
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Connect to the Pinecone index
index = pc.Index(index_name)

# Upsert records in batches
batch_size = 100
records = []
for i, (d, e) in enumerate(zip(email_data, embeddings)):
    records.append({
        'id': d.get('id', f'record_{i}'),
        'values': e,
        'metadata': {
            'source_text': d.get('snippet', ''),
            'sender': d.get('sender', 'unknown'),
            'timestamp': d['timestamp']
        }
    })

    if len(records) >= batch_size or i == len(email_data) - 1:
        index.upsert(vectors=records, namespace='ex-name')
        records = []  # Reset batch

time.sleep(10)  # Wait for upserts to complete
print(index.describe_index_stats())



# Check Redis cache

import time
import json

def cached_query(user_query):
    start_time = time.time()
    
    # Check if the query is already cached in Redis.
    cached_response = redis_client.get(user_query)
    if cached_response:
        elapsed_time = time.time() - start_time
        print("Cache hit! Response retrieved in {:.2f} seconds.".format(elapsed_time))
        return json.loads(cached_response)
    
    print("Cache miss! Generating response...")
    # Time the generation process separately.
    gen_start_time = time.time()
    
    # Generate the embedding for the query.
    xq = client.embeddings.create(input=user_query, model='text-embedding-ada-002').data[0].embedding
    
    # Query Pinecone for similar results.
    results = index.query(
        namespace="ex-name",
        vector=xq,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print('result ',{type(results['matches'])})
    response = json.dumps([item.to_dict()  for item in results['matches']])
    generation_time = time.time() - gen_start_time
    
    # Cache the new response in Redis (cache for 1 hour).
    redis_client.setex(user_query, 3600, response)
    
    total_time = time.time() - start_time
    print("Cache miss! Generation time: {:.2f} seconds, total time: {:.2f} seconds.".format(generation_time, total_time))
    return response

# Example usage:
query = "Google Store"
response = cached_query(query)
print(response)



























# cached_embedding = redis_client.get(query)

# if cached_embedding:
#     print("Cache hit! Using cached embedding.")
#     xq = json.loads(cached_embedding)
# else:
#     print("Cache miss! Generating new embedding.")
#     xq = client.embeddings.create(input=query, model='text-embedding-ada-002').data[0].embedding
#     redis_client.set(query, json.dumps(xq))  # Store in Redis

# # Convert filter timestamp
# filter_date_str = "2025-01-09"
# filter_date_numeric = int(datetime.strptime(filter_date_str, "%Y-%m-%d").timestamp())

# # Perform query in Pinecone with metadata filter
# results = index.query(
#     namespace="ex-name",
#     vector=xq,
#     top_k=3,
#     include_values=False,
#     include_metadata=True
# )

# print(results)
