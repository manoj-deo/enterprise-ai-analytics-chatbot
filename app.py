import json
import math
import os
import random
import sqlite3
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


load_dotenv()

DB_PATH = Path(os.getenv("DATABASE_PATH", "enterprise.db"))
AGENT_MODE = os.getenv("AGENT_MODE", "demo").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)

app = FastAPI(
    title="Enterprise AI Analytics Chatbot",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Request and response schemas
# -------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )


class ChatResponse(BaseModel):
    answer: str
    mode: str
    tools_used: list[str]


# -------------------------------------------------------------------
# Demonstration enterprise data
# -------------------------------------------------------------------


CUSTOMERS = [
    (1, "Acme Retail", "East", 1),
    (2, "Northstar Health", "East", 5),
    (3, "BluePeak Logistics", "South", 2),
    (4, "Vertex Manufacturing", "Midwest", 7),
    (5, "Summit Financial", "West", 1),
    (6, "Greenfield Foods", "South", 4),
    (7, "Atlas Education", "West", 3),
    (8, "Cedar Telecom", "West", 8),
]

PRODUCTS = [
    (1, "Insight Starter", 99.0),
    (2, "Insight Pro", 249.0),
    (3, "Workflow Automator", 399.0),
    (4, "Data Connect", 179.0),
]

DOCUMENTS = {
    "refund_policy": (
        "Customers may request a refund within 30 calendar days "
        "of the initial purchase. Renewal charges are reviewed when "
        "support is contacted within seven days. Refunds over $5,000 "
        "require Finance Operations approval."
    ),
    "support_policy": (
        "Standard support is available Monday through Friday from "
        "8:00 AM to 8:00 PM Eastern Time. Priority-one incidents "
        "have a 30-minute initial-response target. Priority-two "
        "incidents have a four-business-hour response target."
    ),
    "product_guide": (
        "Insight Starter includes dashboards, reports, and CSV export. "
        "Insight Pro provides forecasting, anomaly detection, custom "
        "metrics, and API access. Data Connect supports SQL databases, "
        "REST APIs, and cloud storage."
    ),
}


# -------------------------------------------------------------------
# Database
# -------------------------------------------------------------------


def get_database_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def seed_database() -> None:
    """Create and seed the SQLite database if it is empty."""

    with get_database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                support_tickets INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                FOREIGN KEY (customer_id)
                    REFERENCES customers(id)
            );

            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                FOREIGN KEY (order_id)
                    REFERENCES orders(id),
                FOREIGN KEY (product_id)
                    REFERENCES products(id)
            );
            """
        )

        existing_customers = connection.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]

        if existing_customers:
            return

        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            CUSTOMERS,
        )

        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?)",
            PRODUCTS,
        )

        random_generator = random.Random(42)
        order_id = 1
        item_id = 1

        start_date = date.today() - timedelta(days=360)

        for customer in CUSTOMERS:
            customer_id = customer[0]
            number_of_orders = random_generator.randint(4, 12)

            for _ in range(number_of_orders):
                order_date = start_date + timedelta(
                    days=random_generator.randint(0, 350)
                )

                connection.execute(
                    """
                    INSERT INTO orders
                    VALUES (?, ?, ?)
                    """,
                    (
                        order_id,
                        customer_id,
                        order_date.isoformat(),
                    ),
                )

                selected_products = random_generator.sample(
                    PRODUCTS,
                    random_generator.randint(1, 2),
                )

                for product in selected_products:
                    quantity = random_generator.randint(1, 5)

                    discount_multiplier = random_generator.choice(
                        [1, 1, 0.95, 0.90]
                    )

                    unit_price = round(
                        product[2] * discount_multiplier,
                        2,
                    )

                    connection.execute(
                        """
                        INSERT INTO order_items
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            item_id,
                            order_id,
                            product[0],
                            quantity,
                            unit_price,
                        ),
                    )

                    item_id += 1

                order_id += 1

        connection.commit()


# -------------------------------------------------------------------
# Machine-learning model
# -------------------------------------------------------------------


def train_churn_model():
    """
    Train a demonstration logistic-regression model.

    Features:
    1. Days since last purchase
    2. Orders during the last 90 days
    3. Number of support tickets
    4. Average order value
    """

    random_generator = np.random.default_rng(42)
    sample_count = 800

    recency = random_generator.uniform(
        0,
        240,
        sample_count,
    )

    order_frequency = random_generator.uniform(
        0,
        20,
        sample_count,
    )

    support_tickets = random_generator.poisson(
        3,
        sample_count,
    )

    average_order_value = random_generator.uniform(
        20,
        800,
        sample_count,
    )

    logits = (
        0.03 * recency
        - 0.24 * order_frequency
        + 0.40 * support_tickets
        - 0.001 * average_order_value
        - 2.5
    )

    churn_probabilities = 1 / (
        1 + np.exp(-logits)
    )

    labels = random_generator.binomial(
        1,
        churn_probabilities,
    )

    features = np.column_stack(
        [
            recency,
            order_frequency,
            support_tickets,
            average_order_value,
        ]
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42),
    )

    model.fit(features, labels)

    return model


CHURN_MODEL = train_churn_model()


# -------------------------------------------------------------------
# Agent tools
# -------------------------------------------------------------------


def revenue_summary(
    group_by: str,
    limit: int,
) -> dict:
    """
    Calculate revenue grouped by region, month, or product.
    """

    allowed_groupings = {
        "region": (
            "c.region",
            "region",
        ),
        "month": (
            "substr(o.order_date, 1, 7)",
            "month",
        ),
        "product": (
            "p.name",
            "product",
        ),
    }

    if group_by not in allowed_groupings:
        raise ValueError(
            "group_by must be region, month, or product"
        )

    safe_limit = max(
        1,
        min(int(limit), 20),
    )

    expression, alias = allowed_groupings[group_by]

    query = f"""
        SELECT
            {expression} AS {alias},
            ROUND(
                SUM(i.quantity * i.unit_price),
                2
            ) AS revenue,
            COUNT(DISTINCT o.id) AS orders
        FROM order_items i
        JOIN orders o
            ON o.id = i.order_id
        JOIN customers c
            ON c.id = o.customer_id
        JOIN products p
            ON p.id = i.product_id
        GROUP BY {expression}
        ORDER BY revenue DESC
        LIMIT ?
    """

    with get_database_connection() as connection:
        rows = connection.execute(
            query,
            (safe_limit,),
        ).fetchall()

    return {
        "group_by": group_by,
        "rows": [
            dict(row)
            for row in rows
        ],
    }


def top_customers(limit: int) -> dict:
    """Return customers ranked by total revenue."""

    safe_limit = max(
        1,
        min(int(limit), 20),
    )

    query = """
        SELECT
            c.name AS customer,
            c.region,
            ROUND(
                SUM(i.quantity * i.unit_price),
                2
            ) AS revenue,
            COUNT(DISTINCT o.id) AS orders
        FROM customers c
        JOIN orders o
            ON o.customer_id = c.id
        JOIN order_items i
            ON i.order_id = o.id
        GROUP BY
            c.id,
            c.name,
            c.region
        ORDER BY revenue DESC
        LIMIT ?
    """

    with get_database_connection() as connection:
        rows = connection.execute(
            query,
            (safe_limit,),
        ).fetchall()

    return {
        "rows": [
            dict(row)
            for row in rows
        ]
    }


def tokenize(text: str) -> Counter:
    normalized_text = "".join(
        character.lower()
        if character.isalnum()
        else " "
        for character in text
    )

    return Counter(
        word
        for word in normalized_text.split()
        if len(word) > 2
    )


def calculate_cosine_similarity(
    first_vector: Counter,
    second_vector: Counter,
) -> float:
    common_words = (
        set(first_vector)
        & set(second_vector)
    )

    numerator = sum(
        first_vector[word] * second_vector[word]
        for word in common_words
    )

    first_length = math.sqrt(
        sum(
            value * value
            for value in first_vector.values()
        )
    )

    second_length = math.sqrt(
        sum(
            value * value
            for value in second_vector.values()
        )
    )

    denominator = (
        first_length
        * second_length
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def document_search(
    query: str,
    top_k: int,
) -> dict:
    """Search internal company documents."""

    safe_top_k = max(
        1,
        min(int(top_k), 3),
    )

    query_vector = tokenize(query)
    results = []

    for source, content in DOCUMENTS.items():
        document_vector = tokenize(content)

        score = calculate_cosine_similarity(
            query_vector,
            document_vector,
        )

        if score > 0:
            results.append(
                {
                    "source": source,
                    "content": content,
                    "score": round(score, 4),
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "results": results[:safe_top_k]
    }


def churn_risk(
    limit: int,
    minimum_risk: str,
) -> dict:
    """Predict customer churn using logistic regression."""

    risk_levels = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    if minimum_risk not in risk_levels:
        raise ValueError(
            "minimum_risk must be low, medium, or high"
        )

    safe_limit = max(
        1,
        min(int(limit), 20),
    )

    today = date.today()
    ninety_days_ago = (
        today - timedelta(days=90)
    )

    query = """
        SELECT
            c.name,
            c.support_tickets,
            CAST(
                julianday(?)
                - julianday(MAX(o.order_date))
                AS INTEGER
            ) AS recency,
            COUNT(
                DISTINCT CASE
                    WHEN o.order_date >= ?
                    THEN o.id
                END
            ) AS frequency,
            COALESCE(
                AVG(
                    i.quantity
                    * i.unit_price
                ),
                0
            ) AS order_value
        FROM customers c
        LEFT JOIN orders o
            ON o.customer_id = c.id
        LEFT JOIN order_items i
            ON i.order_id = o.id
        GROUP BY
            c.id,
            c.name,
            c.support_tickets
    """

    with get_database_connection() as connection:
        rows = connection.execute(
            query,
            (
                today.isoformat(),
                ninety_days_ago.isoformat(),
            ),
        ).fetchall()

    predictions = []

    for row in rows:
        feature_values = np.array(
            [
                [
                    float(row["recency"] or 365),
                    float(row["frequency"] or 0),
                    float(row["support_tickets"]),
                    float(row["order_value"] or 0),
                ]
            ]
        )

        probability = float(
            CHURN_MODEL.predict_proba(
                feature_values
            )[0, 1]
        )

        if probability >= 0.70:
            risk_level = "high"
        elif probability >= 0.40:
            risk_level = "medium"
        else:
            risk_level = "low"

        if (
            risk_levels[risk_level]
            >= risk_levels[minimum_risk]
        ):
            predictions.append(
                {
                    "customer": row["name"],
                    "probability": round(
                        probability,
                        4,
                    ),
                    "risk": risk_level,
                }
            )

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "model": "logistic_regression",
        "predictions": predictions[:safe_limit],
        "disclaimer": (
            "This demonstration model was trained "
            "using synthetic data."
        ),
    }


# -------------------------------------------------------------------
# Tool definitions provided to the LLM
# -------------------------------------------------------------------


TOOL_FUNCTIONS = {
    "revenue_summary": revenue_summary,
    "top_customers": top_customers,
    "document_search": document_search,
    "churn_risk": churn_risk,
}


TOOLS = [
    {
        "type": "function",
        "name": "revenue_summary",
        "description": (
            "Get revenue grouped by region, "
            "month, or product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": [
                        "region",
                        "month",
                        "product",
                    ],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": [
                "group_by",
                "limit",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "top_customers",
        "description": (
            "Rank customers by total revenue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                }
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "document_search",
        "description": (
            "Search internal policies and "
            "product documentation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                },
            },
            "required": [
                "query",
                "top_k",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "churn_risk",
        "description": (
            "Predict customer churn risk using "
            "a logistic-regression model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
                "minimum_risk": {
                    "type": "string",
                    "enum": [
                        "low",
                        "medium",
                        "high",
                    ],
                },
            },
            "required": [
                "limit",
                "minimum_risk",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def call_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    function = TOOL_FUNCTIONS.get(
        tool_name
    )

    if function is None:
        return {
            "error": (
                f"Unknown tool: {tool_name}"
            )
        }

    try:
        return function(**arguments)
    except (
        TypeError,
        ValueError,
    ) as error:
        return {
            "error": str(error)
        }


# -------------------------------------------------------------------
# OpenAI agent
# -------------------------------------------------------------------


def run_openai_agent(
    message: str,
) -> tuple[str, list[str]]:
    if openai_client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    input_items = [
        {
            "role": "user",
            "content": message,
        }
    ]

    tools_used = []

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are an enterprise AI analytics assistant. "
            "Use tools whenever an answer depends on company data. "
            "Never invent business metrics or policy information. "
            "Explain results clearly and mention that the dataset "
            "and churn model are demonstrations."
        ),
        input=input_items,
        tools=TOOLS,
    )

    for _ in range(5):
        function_calls = [
            output
            for output in response.output
            if output.type == "function_call"
        ]

        if not function_calls:
            return (
                response.output_text
                or "No answer was generated.",
                tools_used,
            )

        # Preserve the model's function-call output.
        input_items += response.output

        for function_call in function_calls:
            arguments = json.loads(
                function_call.arguments
            )

            tool_result = call_tool(
                function_call.name,
                arguments,
            )

            tools_used.append(
                function_call.name
            )

            input_items.append(
                {
                    "type": (
                        "function_call_output"
                    ),
                    "call_id": (
                        function_call.call_id
                    ),
                    "output": json.dumps(
                        tool_result
                    ),
                }
            )

        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=(
                "Explain the supplied tool results "
                "in concise business language."
            ),
            input=input_items,
            tools=TOOLS,
        )

    return (
        "The agent reached its tool-call limit.",
        tools_used,
    )


# -------------------------------------------------------------------
# Demo agent
# -------------------------------------------------------------------


def run_demo_agent(
    message: str,
) -> tuple[str, list[str]]:
    """
    Fallback agent that works without an API key.
    """

    normalized_message = message.lower()

    if any(
        word in normalized_message
        for word in [
            "refund",
            "support",
            "policy",
        ]
    ):
        result = document_search(
            query=message,
            top_k=2,
        )

        if not result["results"]:
            return (
                "No matching document was found.",
                ["document_search"],
            )

        answer = "\n\n".join(
            (
                f"{item['source']}: "
                f"{item['content']}"
            )
            for item in result["results"]
        )

        return (
            answer,
            ["document_search"],
        )

    if any(
        word in normalized_message
        for word in [
            "churn",
            "risk",
            "retention",
        ]
    ):
        minimum_risk = (
            "high"
            if "high" in normalized_message
            else "medium"
        )

        result = churn_risk(
            limit=5,
            minimum_risk=minimum_risk,
        )

        predictions = result["predictions"]

        if not predictions:
            return (
                "No customers matched that risk threshold.",
                ["churn_risk"],
            )

        lines = [
            (
                f"- {prediction['customer']}: "
                f"{prediction['probability']:.1%} "
                f"({prediction['risk']} risk)"
            )
            for prediction in predictions
        ]

        answer = (
            "Customers with elevated churn risk:\n\n"
            + "\n".join(lines)
            + "\n\nThis model uses synthetic "
            "demonstration data."
        )

        return (
            answer,
            ["churn_risk"],
        )

    if (
        "top" in normalized_message
        and "customer" in normalized_message
    ):
        result = top_customers(limit=5)

        lines = [
            (
                f"- {row['customer']} "
                f"({row['region']}): "
                f"${row['revenue']:,.2f} "
                f"across {row['orders']} orders"
            )
            for row in result["rows"]
        ]

        return (
            "Top customers:\n\n"
            + "\n".join(lines),
            ["top_customers"],
        )

    if (
        "month" in normalized_message
        or "trend" in normalized_message
    ):
        group_by = "month"
    elif "product" in normalized_message:
        group_by = "product"
    else:
        group_by = "region"

    result = revenue_summary(
        group_by=group_by,
        limit=12,
    )

    lines = [
        (
            f"- {row[group_by]}: "
            f"${row['revenue']:,.2f} "
            f"from {row['orders']} orders"
        )
        for row in result["rows"]
    ]

    return (
        f"Revenue grouped by {group_by}:\n\n"
        + "\n".join(lines)
        + "\n\nThis response uses seeded "
        "demonstration data.",
        ["revenue_summary"],
    )


# -------------------------------------------------------------------
# API routes
# -------------------------------------------------------------------


@app.on_event("startup")
def startup_event() -> None:
    seed_database()


@app.get("/health")
def health() -> dict:
    actual_mode = (
        "openai"
        if AGENT_MODE == "openai"
        and openai_client
        else "demo"
    )

    return {
        "status": "healthy",
        "mode": actual_mode,
        "model": OPENAI_MODEL,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    try:
        if (
            AGENT_MODE == "openai"
            and openai_client
        ):
            answer, tools_used = (
                run_openai_agent(
                    request.message
                )
            )

            mode = "openai"
        else:
            answer, tools_used = (
                run_demo_agent(
                    request.message
                )
            )

            mode = "demo"

        return ChatResponse(
            answer=answer,
            mode=mode,
            tools_used=tools_used,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


# -------------------------------------------------------------------
# Simple frontend
# -------------------------------------------------------------------


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">

    <meta
        name="viewport"
        content="width=device-width"
    >

    <title>
        Enterprise AI Analytics Agent
    </title>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background: #07131d;
            color: #eef5f7;
        }

        .container {
            max-width: 900px;
            margin: auto;
            padding: 40px 20px;
        }

        h1 {
            color: #70d5cb;
        }

        .subtitle {
            color: #a5b5c0;
        }

        .examples {
            margin: 24px 0;
        }

        .examples button {
            margin: 4px;
            padding: 10px;
            background: #18303f;
            color: white;
            border: 1px solid #345;
            border-radius: 8px;
            cursor: pointer;
        }

        #chat {
            min-height: 420px;
            max-height: 550px;
            overflow-y: auto;
            padding: 20px;
            background: #0e2230;
            border-radius: 14px;
        }

        .message {
            padding: 12px;
            margin: 10px 0;
            border-radius: 10px;
            white-space: pre-wrap;
        }

        .user {
            margin-left: 20%;
            background: #174149;
        }

        .assistant {
            margin-right: 20%;
            background: #172b39;
        }

        form {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        input {
            flex: 1;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #345;
            background: #102531;
            color: white;
        }

        form button {
            padding: 12px 18px;
            border: 0;
            border-radius: 10px;
            background: #70d5cb;
            font-weight: bold;
            cursor: pointer;
        }

        small {
            color: #9badb7;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>
            Enterprise AI Analytics Agent
        </h1>

        <p class="subtitle">
            SQL analytics, document retrieval,
            and customer churn prediction.
        </p>

        <div class="examples">
            <button onclick="askQuestion(this.innerText)">
                Revenue by region
            </button>

            <button onclick="askQuestion(this.innerText)">
                Show monthly revenue trends
            </button>

            <button onclick="askQuestion(this.innerText)">
                Who are the top customers?
            </button>

            <button onclick="askQuestion(this.innerText)">
                Which customers have high churn risk?
            </button>

            <button onclick="askQuestion(this.innerText)">
                What is the refund policy?
            </button>
        </div>

        <div id="chat">
            <div class="message assistant">
                Ask me a business question.
            </div>
        </div>

        <form id="chat-form">
            <input
                id="message-input"
                placeholder="Ask a question..."
                required
            >

            <button type="submit">
                Send
            </button>
        </form>
    </div>

    <script>
        const form = document.querySelector(
            "#chat-form"
        );

        const input = document.querySelector(
            "#message-input"
        );

        const chat = document.querySelector(
            "#chat"
        );

        function addMessage(
            message,
            role,
            tools = []
        ) {
            const container =
                document.createElement("div");

            container.className =
                `message ${role}`;

            const text =
                document.createElement("div");

            text.textContent = message;

            container.appendChild(text);

            if (tools.length > 0) {
                const toolText =
                    document.createElement("small");

                toolText.textContent =
                    `Tools used: ${tools.join(", ")}`;

                container.appendChild(
                    document.createElement("br")
                );

                container.appendChild(toolText);
            }

            chat.appendChild(container);
            chat.scrollTop = chat.scrollHeight;
        }

        async function askQuestion(message) {
            if (!message) {
                return;
            }

            addMessage(
                message,
                "user"
            );

            try {
                const response = await fetch(
                    "/chat",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body: JSON.stringify({
                            message: message
                        })
                    }
                );

                const data =
                    await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.detail
                        || "Request failed"
                    );
                }

                addMessage(
                    data.answer,
                    "assistant",
                    data.tools_used
                );
            } catch (error) {
                addMessage(
                    `Error: ${error.message}`,
                    "assistant"
                );
            }
        }

        form.addEventListener(
            "submit",
            async function (event) {
                event.preventDefault();

                const message =
                    input.value.trim();

                input.value = "";

                await askQuestion(message);
            }
        );
    </script>
</body>
</html>
"""
