from generator import prompt_to_query 
from openai import OpenAI
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import openai
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
            {"role": "system", "content": "You are an assistant that summarizes email threads."},
            {"role": "user", "content": f"Summarize the following email thread:\n\n{thread_emails}"}
        ],
        max_tokens=100,
        temperature=0.7
    )

    print(response)
    summary = response.choices[0].message.content.strip()
    return summary

def fetch_emails_from_neo4j(driver, cypher_query):
    """
    Execute a Cypher query and fetch email data from Neo4j.
    Args:
        driver: Neo4j database driver.
        cypher_query: Cypher query string.

    Returns:
        list: List of email dictionaries.
    """
    with driver.session() as session:
        result = session.run(cypher_query)
        # @verify the result
        # print(result)
        emails = []
        for record in result:
            # print("sample from the records fetched from the database: ", record)
            emails.append(record) 
        # print(emails)
        return emails

prompt = 'summarize the emails where the sender is "Google Store <googlestore-noreply@google.com>"' 

cypher_query = prompt_to_query(user_prompt=prompt, schema=gr_schema, api_key=openai_api_key)
print("*********** CYPHER QUERY GENERATED ****************")
email_threads = fetch_emails_from_neo4j(driver=driver, cypher_query=cypher_query)
print('*********** THE DATA FETCHED FROM THE DATABASE ****************')
summary = summarize_thread_with_openai(email_threads)

print("Thread Summary:", summary)