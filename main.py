import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from typing import List

load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# Pydantic models for requests
class DBCredentials(BaseModel):
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    db_name: str


class QueryRequest(BaseModel):
    question: str
    db_credentials: DBCredentials


class DBStructureRequest(BaseModel):
    db_credentials: DBCredentials


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    db_credentials: DBCredentials


# Function to convert natural language to SQL
def nl_to_sql(question: str, table_info: str) -> str:
    prompt = f"""
    Given the following tables in a PostgreSQL database:

    {table_info}

    Convert the following natural language question to a SQL query:

    {question}

    Return only the SQL query, without any additional explanation.
    Also the sql code should not have ``` in beginning or end and sql word in output
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert. Convert natural language questions to SQL queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def get_db_structure(db_credentials: DBCredentials):
    db_url = f"postgresql://{db_credentials.db_user}:{db_credentials.db_password}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
    engine = create_engine(db_url)
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public';"))
        table_info = pd.DataFrame(result.fetchall(), columns=result.keys())
    return table_info.groupby('table_name')['column_name'].apply(list).to_dict()


def execute_sql_query(query: str, db_credentials: DBCredentials):
    db_url = f"postgresql://{db_credentials.db_user}:{db_credentials.db_password}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
    engine = create_engine(db_url)
    with engine.connect() as connection:
        result = connection.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df.to_dict(orient="records")


def format_response_with_llm(sql_query: str, query_results: str) -> str:
    prompt = f"""
    Analyze the following query results and provide insights:

    Results: {query_results}

    Please provide a clear and concise analysis of the data. Focus on key trends, patterns, or notable information in the results. Use markdown formatting to structure your response, including:

    - Headers for main sections
    - Bullet points or numbered lists for key points
    - Bold or italic text for emphasis
    - Code blocks for any numerical data or examples

    Your analysis should be informative and easy to understand for someone looking at this data.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a data analyst providing insights on query results. Use markdown formatting in your responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    formatted_response = response.choices[0].message.content.strip()

    # Add the SQL query at the end without displaying it in the chat
    formatted_response += f"\n\n[SQL_QUERY]{sql_query}[/SQL_QUERY]"

    return formatted_response

# def format_response_with_llm(sql_query: str, query_results: str) -> str:
#     prompt = f"""
#     Given the following SQL query and its results, provide a natural language explanation:
#
#     SQL Query: {sql_query}
#
#     Results: {query_results}
#
#     Please explain the results in a clear, concise manner. At the end, include the original SQL query in a 'card' format.
#     """
#
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system",
#              "content": "You are a helpful AI assistant that explains SQL query results in natural language."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7
#     )
#
#     formatted_response = response.choices[0].message.content.strip()
#     formatted_response += f"\n\n---\nExecuted Query:\n```sql\n{sql_query}\n```"
#
#     return formatted_response


@app.post("/query")
async def query(request: QueryRequest):
    try:
        db_url = f"postgresql://{request.db_credentials.db_user}:{request.db_credentials.db_password}@{request.db_credentials.db_host}:{request.db_credentials.db_port}/{request.db_credentials.db_name}"
        engine = create_engine(db_url)

        # Get table information
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public';"))
            table_info = pd.DataFrame(result.fetchall(), columns=result.keys())

        table_info_str = table_info.groupby('table_name')['column_name'].apply(list).to_dict()
        table_info_str = "\n".join(
            [f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in table_info_str.items()])

        # Convert natural language to SQL
        sql_query = nl_to_sql(request.question, table_info_str)

        # Execute the SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Convert results to JSON
        json_result = df.to_dict(orient="records")

        return {
            "question": request.question,
            "sql_query": sql_query,
            "results": json_result
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        db_structure = get_db_structure(request.db_credentials)

        system_message = f"""You are a helpful AI assistant that can query a PostgreSQL database. 
        Here's the database schema: {db_structure}
        When generating SQL queries, do not include ``` or 'sql' tags. Only return the raw SQL query."""

        messages = [{"role": "system", "content": system_message}] + [m.dict() for m in request.messages]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        ai_message = response.choices[0].message.content.strip()

        # Check if the AI's response contains a SQL query
        if "SELECT" in ai_message.upper():
            try:
                results = execute_sql_query(ai_message, request.db_credentials)
                formatted_response = format_response_with_llm(ai_message, str(results))

                # Include the tabular data in the response
                return {
                    "role": "assistant",
                    "content": formatted_response,
                    "tabular_data": results  # This is the new addition
                }
            except Exception as e:
                error_message = f"Error executing query: {str(e)}"
                formatted_response = format_response_with_llm(ai_message, error_message)
                return {"role": "assistant", "content": formatted_response}
        else:
            return {"role": "assistant", "content": ai_message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db-structure")
async def get_db_structure_endpoint(request: DBStructureRequest):
    try:
        structure = get_db_structure(request.db_credentials)
        return {"structure": structure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))