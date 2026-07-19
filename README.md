
# Enterprise AI Analytics Chatbot

An enterprise-style AI agent that lets users ask natural-language questions
across structured business data, internal documents, and a machine-learning
model.

## Features

- OpenAI function calling
- FastAPI backend
- SQLite enterprise database
- Revenue analytics
- Customer ranking
- Internal-document retrieval
- Customer churn prediction
- Scikit-learn logistic regression
- Tool execution traces
- Browser-based chatbot interface
- Demo mode that works without an API key

## Example questions

- What was revenue by region?
- Show monthly revenue trends.
- Which products generated the most revenue?
- Who are the top five customers?
- Which customers have high churn risk?
- What is the refund policy?
- What is the support response time?

## Architecture

```text
Browser
   |
FastAPI Chat Endpoint
   |
AI Agent
   |----------------------|----------------------|
Revenue Analytics     Document Retrieval     Churn Prediction
   |                      |                      |
SQLite Database       Company Policies      Logistic Regression
