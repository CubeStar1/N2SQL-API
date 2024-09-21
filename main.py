import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Initialize OpenAI client
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


@app.post("/query")
async def query(request: QueryRequest):
    try:
        print(request.db_credentials.dict())
        # Create SQLAlchemy engine with provided credentials
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

@app.post("/db-structure")
async def get_db_structure(request: DBStructureRequest):
    try:
        # Create SQLAlchemy engine with provided credentials
        db_url = f"postgresql://{request.db_credentials.db_user}:{request.db_credentials.db_password}@{request.db_credentials.db_host}:{request.db_credentials.db_port}/{request.db_credentials.db_name}"
        engine = create_engine(db_url)

        # Get table information
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public';"))
            table_info = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Structure the data
        structure = table_info.groupby('table_name')['column_name'].apply(list).to_dict()

        return {"structure": structure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

