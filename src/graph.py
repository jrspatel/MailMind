from neo4j import GraphDatabase 
import json 
from textblob import TextBlob
import os
from dotenv import load_dotenv
load_dotenv() 

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri=uri, auth=(username, password)) 



# import dotenv
# import os
# from neo4j import GraphDatabase

# load_status = dotenv.load_dotenv("Neo4j-a0a2fa1d-Created-2023-11-06.txt")
# if load_status is False:
#     raise RuntimeError('Environment variables not loaded.')

# URI = os.getenv("NEO4J_URI")
# AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# with GraphDatabase.driver(URI, auth=AUTH) as driver:
#     driver.verify_connectivity()
#     print("Connection established.")


def analyze_sentiment(snippet):
    analysis = TextBlob(snippet)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)


def insert_email(tx, email):

    sentiment = analyze_sentiment(email['snippet'])
    print(sentiment)
    query = """
        MERGE (sender:Person {email: $sender})
        MERGE (receiver:Person {email: $receiver})
        MERGE (email:Email {
                        id: $id,
                        subject: $subject,
                        snippet: $snippet,
                        timestamp: $timestamp,
                        sentiment: $sentiment
                })
        MERGE (sender)-[r:SENT]->(email)
        MERGE (email)-[r2:RECEIVED]->(receiver)

        ON CREATE SET r.freq = 1, r2.freq = 1
        ON MATCH SET r.freq = r.freq + 1, r2.freq = r2.freq + 1
        """
    
    tx.run(query, 
           sender=email["sender"], 
           receiver=email["receiver"], 
           id=email["id"], 
           subject=email["subject"], 
           snippet=email["snippet"], 
           timestamp=email["timestamp"],
           sentiment=sentiment)
    
     
def load_json_to_neo4j(json_file_path, driver):
    # load json data 
    with open(json_file_path, 'r') as f:
        email_data = json.load(f) 
    
    # insert the data 
    with driver.session() as session:
        for email in email_data:
            print(email) 
            
            session.write_transaction(insert_email, email)
    print('data loaded successfully')

json_file_path = 'D:\GMAIL\emails.json'
load_json_to_neo4j(json_file_path, driver)

# try:
#     with driver.session() as session:
#         result = session.run("RETURN 'Connection Successful' AS message")
#         for record in result:
#                 print(record["message"]) 
# except:
#      print(" connection was not successful...")


"""
    Represent edges:
        Frequency of communication.
        Sentiment scores, shared topics, or time-based connections.
"""