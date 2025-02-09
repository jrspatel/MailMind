from generator import prompt_to_query 
from openai import OpenAI
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import openai
import numpy as np
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri=uri, auth=(username, password)) 
graph = Neo4jGraph(url=uri, username=username, password=password)

# Fetch the schema (assumes graphdatascience is initialized)
gr_schema = graph.schema




def summarize_thread_with_openai(thread_emails):
    # Combine the snippets of all emails in the thread

    # print("thread emails", thread_emails)

    # thread_text = "\n".join([email["snippet"] for email in thread_emails])
    
    
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a sophisticated assistant designed to assist with emails by summarizing content, retrieving relevant information, generating responses, and automating workflows." 
                                         "Your actions are based on the specific use-case requirements provided, focusing on accuracy, context-awareness, and efficiency."},
            {"role": "user", "content": f"Summarize the following message\n\n{thread_emails}"}
        ],
        max_tokens=100,
        temperature=0.7
    )

    # print(response)
    summary = response.choices[0].message.content.strip()
    return summary



def regenerate_query_with_error(orginal_query, error_trace, schema):
    """
        A function which regenerates a cypher query based on the error trace
    """

    prompt = f"""
                The following Cypher query failed with this error trace: "{error_trace}". 
                Here's the original query: "{orginal_query}". 
                The schema of the database: "{schema}".
                Please modify the query to correct the error and make it syntactically valid.
        """
        # Here, call your AI model to regenerate the query or apply logic to fix it.
    regenerated_query = prompt_to_query(prompt, gr_schema, api_key=openai_api_key)  # Assuming send_to_prompt is a function that handles prompt submission.
    return regenerated_query



def fetch_emails_from_neo4j(driver, cypher_query, retries_left, gr_schema):
    """
    Execute a Cypher query and fetch email data from Neo4j.
    Args:
        driver: Neo4j database driver.
        cypher_query: Cypher query string.

    Returns:
        list: List of email dictionaries.
    """

    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            print(result)
            emails = []
            for record in result:
                # print("sample from the records fetched from the database: ", record)
                emails.append(record) 
            print(emails)
            

            if emails == []:
                print("No emails found. Regenerating query.")
                regenerate_query = regenerate_query_with_error(cypher_query, "No emails found.", schema=gr_schema)
                print(f"Executing the re-generated query: {regenerate_query}") 
                if retries_left>0:
                    regenerate_query = regenerate_query_with_error(cypher_query, trace, schema=gr_schema) 

                    print(f"Executing the re-generated query: {regenerate_query}") 
                    return fetch_emails_from_neo4j(driver= driver, cypher_query= regenerate_query, retries_left= retries_left - 1, gr_schema = gr_schema) 
                else :
                    # if retries maxed out 
                    print(" No retries left !!")
                    return []
        
        return emails

    except Exception as e:
        trace = str(e) 
        print(f"Executing the error messgae: {trace}") 
        if retries_left>0:
            regenerate_query = regenerate_query_with_error(cypher_query, trace, schema=gr_schema) 

            print(f"Executing the re-generated query: {regenerate_query}") 
            return fetch_emails_from_neo4j(driver= driver, cypher_query= regenerate_query, retries_left= retries_left - 1, gr_schema = gr_schema) 
        else :
            # if retries maxed out 
            print(" No retries left !!")
            return []
        

def evaluation(prompt, response):
    """
        converting the user prompt to embeddings - understand the user intent.
        convert the response into embeddings.

        calculate the simmilartiy between these embeddings.
    """
    prompt_embed = openai.embeddings.create(
        input= prompt,
        model= "text-embedding-3-small"
    ).data[0].embedding
    
    response_embed = openai.embeddings.create(
        input= response,
        model= "text-embedding-3-small"
    ).data[0].embedding
    
    prompt_vector = np.array(prompt_embed)
    response_vector = np.array(response_embed)

    # Calculate cosine similarity
    similarity = np.dot(prompt_vector, response_vector) / (
        np.linalg.norm(prompt_vector) * np.linalg.norm(response_vector)
    )

    print(similarity)


prompt = 'summarize the emails where the sender is "Googjole Store <googlestore-noreply@google.com>" nnihi' 

#prompt = 'summarize the emails i got yesterday {2025-1-24}'

# cypher_query = prompt_to_query(user_prompt=prompt, schema=gr_schema, api_key=openai_api_key)
# print("*********** CYPHER QUERY GENERATED ****************")
# email_threads = fetch_emails_from_neo4j(driver=driver, cypher_query=cypher_query, retries_left= 2, gr_schema= gr_schema)


# print('*********** THE DATA FETCHED FROM THE DATABASE ****************')
# summary = summarize_thread_with_openai(email_threads)

# print("Thread Summary:", summary)

# print("********* Evaluation in Progress *******************")
# evaluation(prompt=prompt, response= summary) 

from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1,
)

print(response.data[0].url)