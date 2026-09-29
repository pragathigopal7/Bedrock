# Codebase Q&A Agent on Amazon Bedrock AgentCore

A proof of concept agent that answers questions about a Java Spring Boot codebase, cites file and line references, and refuses to guess. It is hosted on **Amazon Bedrock AgentCore Runtime** and built with the **Strands Agents SDK**. The point of the project is the accuracy measurement around it, not just the agent.

## Use case

Engineers modernizing or migrating a legacy Java service spend hours finding where behavior lives (retries, idempotency, event publishing). This agent retrieves the relevant code, answers from it, and cites `path:start-end` for every claim so a reviewer can verify in seconds.

## Architecture

```
Client (AWS SDK / CLI)  ->  AgentCore Runtime  ->  Strands agent  ->  Foundation model on Amazon Bedrock
                                                        |
                                            tools: search_code, read_file
                                                        |
                                     retriever.py (method aware chunks + BM25)
                                                        |
                                                  sample_repo/
```

| Piece | What it does |
|---|---|
| `agent.py` | Strands agent wrapped with `BedrockAgentCoreApp`; `@app.entrypoint` accepts `{"prompt": "..."}` |
| `retriever.py` | Chunks code at method boundaries, ranks with BM25 (no external dependencies) |
| Tool call budget | Caps tool calls per question (default 8) so the agent cannot loop or run up cost |
| Path guard | `read_file` refuses any path outside the repo root |
| System prompt | Answer only from retrieved code, cite every claim, say "Not found in the code." otherwise |
| `eval/` | Golden set of 11 questions plus a harness that scores retrieval and answers |

## Measuring accuracy

The golden set has 10 answerable questions with expected source files and expected facts, plus 1 deliberately unanswerable question to test that the agent refuses instead of hallucinating.

| Layer | Metric | How |
|---|---|---|
| Retrieval | recall@k, MRR | Did the right file appear in the top k results? |
| Answer | fact coverage | Does the answer contain the expected facts? |
| Safety | refusal rate | Does it say the code lacks the answer for the unanswerable question? |

Retrieval baseline on the sample repo (run locally, no AWS needed):

```
python eval/run_eval.py --k 3
Retrieval: recall@3 = 100%, MRR = 0.88 over 10 questions
```

The MRR below 1.0 shows where ranking is imperfect (the timeout and Kafka producer questions rank the right file 3rd and 2nd). That is the motivation for the next step below.

Answer accuracy requires Bedrock access: `python eval/run_eval.py --answers`. Record your own numbers in the table before quoting them anywhere.

## Run it

Prerequisites: Python 3.10+, AWS credentials, and access to a foundation model enabled in Amazon Bedrock in your region.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt bedrock-agentcore-starter-toolkit

export MODEL_ID=<a Bedrock model ID enabled in your account>

# Local
python agent.py
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How are retries configured for the payment service?"}'

# Deploy to AgentCore Runtime
agentcore configure --entrypoint agent.py
agentcore launch
agentcore invoke '{"prompt": "What happens when the payment circuit breaker opens?"}'
```

Point it at your own repo with `REPO_ROOT=/path/to/repo`.

## Next steps

1. **Semantic retrieval:** add Amazon Titan embeddings and a vector store, then compare recall@k and MRR against the BM25 baseline using the same golden set.
2. **AgentCore Gateway:** expose repo search as a tool through Gateway so other agents can reuse it.
3. **AgentCore Memory:** keep conversation context across a code review session.
4. **AgentCore Observability:** trace every tool call and track latency and cost per question.
5. **LLM as judge:** add a rubric based judge for answers where keyword matching is too strict, and calibrate it against human review.
6. **Change proposals:** generate refactors as pull requests, gated by compile, tests, and human review.
