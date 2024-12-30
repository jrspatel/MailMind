import openai

def prompt_to_query(user_prompt, schema, api_key):
    
    """
    Converts a user prompt into a Cypher query using a provided schema.

    Args:
        user_prompt (str): Natural language query from the user.
        schema (str): Schema definition of the Neo4j database.
        client (OpenAI): OpenAI client for query generation.

    Returns:
        str: Generated Cypher query.
    """

    openai.api_key = api_key

    CYPHER_GENERATION_TEMPLATE = """
    Task: Generate a Cypher statement to query a graph database.
    Instructions:
    - Use only the provided relationship types and properties in the schema.
    - Do not use any other relationship types or properties that are not provided.
    - Ensure the generated Cypher query includes a RETURN statement explicitly listing 
      property values used in filtering conditions and the main information requested.
    
    Schema:
    {schema}

    The question is:
    {question}

    Note:
    - Do not include any explanations or apologies in your responses.
    - Respond only with the generated Cypher statement.
    """

    # Populate the prompt with schema and question
    formatted_prompt = CYPHER_GENERATION_TEMPLATE.format(schema=schema, question=user_prompt)

    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant that translates natural language "
            "queries into Cypher queries for Neo4j, strictly adhering to the provided schema."
        )
    }
    user_message = {
        "role": "user",
        "content": formatted_prompt
    }

    # Use OpenAI to generate the Cypher query
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" depending on your needs
        messages=[system_message, user_message],
        max_tokens=200
    )

    # Extract the generated Cypher query
    cypher_query = response.choices[0].message.content.strip()
    print("Generated Cypher Query:")
    print(cypher_query)
    return cypher_query
