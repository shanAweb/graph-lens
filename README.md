# GraphLens — Knowledge Graph + Semantic Search

A hybrid retrieval system that fuses a **Neo4j knowledge graph** with a **Qdrant
vector store** behind a single **FastAPI** service. Documents are ingested,
cleaned, run through NLP (NER + relation extraction + co-reference), and split
into two indexes simultaneously: structured triples land in Neo4j, embedded
chunks land in Qdrant. At query time, semantic search retrieves the most
relevant passages and the graph layer enriches them with related entities and
paths — giving you both the *snippet* and the *connections* in one response.

---

## Table of Contents

1. [Why this project](#why-this-project)
2. [Architecture](#architecture)
3. [Architecture flow](#architecture-flow)
4. [Directory structure](#directory-structure)
5. [Module-by-module breakdown](#module-by-module-breakdown)
6. [Core libraries](#core-libraries)
7. [Neo4j schema](#neo4j-schema)
8. [API reference](#api-reference)
9. [Environment configuration](#environment-configuration)
10. [Docker setup](#docker-setup)
11. [Step-by-step build plan](#step-by-step-build-plan)
12. [Running locally](#running-locally)
13. [Testing](#testing)
14. [Demo dataset & sample queries](#demo-dataset--sample-queries)
15. [Roadmap](#roadmap)

---

## Why this project

Pure vector search returns *similar text*. Pure graph search returns *related
entities*. Real questions need both — "Which companies has Elon Musk founded,
and what are they working on?" is half-graph (founded-relation traversal),
half-semantic ("working on" is fuzzy and lives in prose).

GraphLens combines them:

- **Vector layer** — finds passages that match the *meaning* of a query.
- **Graph layer** — finds entities and relationships that match the *structure*
  of a query.
- **Hybrid layer** — runs vector search first, lifts the entities mentioned in
  the top chunks, expands them in the graph, and returns a unified result.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                            │
│   Raw text (PDF, articles, docs)  ·  Web scrape (Wikipedia,     │
│   news APIs)  ·  Structured data (CSV, JSON, databases)         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                           NLP LAYER                             │
│   NER (spaCy · en_core_web_trf)                                 │
│   Relation extraction (Rebel · HuggingFace)                     │
│   Co-reference resolution (coreferee)                           │
│   Chunking (clean · split · 512-token overlap)                  │
└──────────────┬──────────────────────────┬───────────────────────┘
               ▼                          ▼
┌───────────────────────────┐  ┌───────────────────────────────┐
│   GRAPH STORE — Neo4j     │  │   VECTOR STORE — Qdrant       │
│   Nodes: Person, Org,     │  │   Text chunk embeddings       │
│          Place, Concept,  │  │   Payload: source, entity refs│
│          Document         │  │   Metric: cosine              │
│   Edges: WORKS_AT,        │  │                               │
│          RELATED_TO,      │  │                               │
│          PART_OF,         │  │                               │
│          LOCATED_IN,      │  │                               │
│          MENTIONS         │  │                               │
└──────────────┬────────────┘  └───────────────┬───────────────┘
               │                               ▲
               │                               │
               │           ┌───────────────────┴────────────┐
               │           │  EMBEDDINGS                    │
               │           │  sentence-transformers         │
               │           │  all-MiniLM-L6-v2 (384-dim)    │
               │           └────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVE LAYER                             │
│   FastAPI — search API                                          │
│   POST /search · GET /entity/{name} · GET /graph/{entity}       │
│   GET /health                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Layers, colour-coded as in the diagram:**

| Layer       | Component                       | Purpose                                         |
|-------------|---------------------------------|-------------------------------------------------|
| Input       | `ingestion/`                    | Pull raw content from PDFs, web, structured src |
| NLP         | `nlp/`                          | Entities, relations, co-reference, chunking     |
| Graph store | `graph/` → Neo4j                | Structured knowledge graph                      |
| Vector store| `vectorstore/` → Qdrant         | Embedded chunks with rich payload               |
| Embeddings  | `vectorstore/embedder.py`       | sentence-transformers → 384-d vectors           |
| Serve       | `api/`                          | FastAPI endpoints for search, entity, graph     |

---

## Architecture flow

End-to-end, a document moves through the system like this:

1. **Ingest.** `ingestion/file_loader.py` (PDFs/TXTs/JSONs) or
   `ingestion/web_scraper.py` (Wikipedia/news) pulls raw bytes.
   `ingestion/cleaner.py` strips HTML, normalises whitespace, drops noise.

2. **NLP.** Clean text fans out into three parallel passes:
   - `nlp/ner.py` extracts named entities via spaCy `en_core_web_trf`.
   - `nlp/coreference.py` resolves pronouns ("he", "the company") to their
     antecedent so downstream relations attach to the right entity.
   - `nlp/relation_extractor.py` runs the **Rebel** seq2seq model and emits
     `(head, relation, tail)` triples.
   - In parallel, `nlp/chunker.py` slices the cleaned text into overlapping
     512-token chunks (configurable overlap, default 64).

3. **Write to graph.** `graph/graph_builder.py` converts entities + triples
   into Cypher `MERGE` statements via `graph/neo4j_client.py`. Re-running the
   pipeline on the same source is idempotent — a node for "SpaceX" exists once,
   regardless of how many documents mention it.

4. **Write to vector store.** `vectorstore/embedder.py` encodes each chunk to a
   384-d vector with `all-MiniLM-L6-v2`. `vectorstore/indexer.py` batch-upserts
   chunks into the Qdrant collection with payload `{source, chunk_index,
   entities, text}`.

5. **Query.** A user POSTs a natural-language query to `/search`:
   - `search/semantic_search.py` embeds the query and pulls top-*k* chunks
     from Qdrant.
   - `search/graph_search.py` takes entities surfaced in those chunks and
     traverses Neo4j for neighbours and relationships.
   - `search/hybrid_search.py` merges both views into a single response —
     ranked chunks **plus** their graph context.

6. **Serve.** `api/main.py` mounts the four routes under FastAPI. Pydantic
   schemas in `api/schemas.py` validate every request and response.

---

## Directory structure

```
knowledge-graph-semantic-search/
│
├── data/
│   ├── raw/                        # Raw input files (PDFs, TXTs, JSONs)
│   ├── processed/                  # Cleaned & chunked text
│   └── samples/                    # Sample datasets for demo
│
├── ingestion/
│   ├── __init__.py
│   ├── file_loader.py              # Load PDFs, TXTs, CSVs
│   ├── web_scraper.py              # Wikipedia / news API scraper
│   └── cleaner.py                  # Text cleaning & preprocessing
│
├── nlp/
│   ├── __init__.py
│   ├── ner.py                      # Named Entity Recognition (spaCy)
│   ├── relation_extractor.py       # Relation extraction (Rebel model)
│   ├── coreference.py              # Co-reference resolution
│   └── chunker.py                  # Text chunking with overlap
│
├── graph/
│   ├── __init__.py
│   ├── neo4j_client.py             # Neo4j connection & session manager
│   ├── graph_builder.py            # Build nodes & edges from NLP output
│   ├── graph_queries.py            # Cypher query templates
│   └── schema.py                   # Node/edge type definitions
│
├── vectorstore/
│   ├── __init__.py
│   ├── embedder.py                 # Sentence-transformer embeddings
│   ├── qdrant_client.py            # Qdrant connection & collection setup
│   └── indexer.py                  # Batch upsert chunks into Qdrant
│
├── search/
│   ├── __init__.py
│   ├── semantic_search.py          # Vector similarity search (Qdrant)
│   ├── graph_search.py             # Graph traversal queries (Neo4j)
│   └── hybrid_search.py            # Combine vector + graph results
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point
│   ├── routes/
│   │   ├── search.py               # POST /search
│   │   ├── entity.py               # GET /entity/{name}
│   │   ├── graph.py                # GET /graph/{entity}
│   │   └── health.py               # GET /health
│   └── schemas.py                  # Pydantic request/response models
│
├── pipeline/
│   ├── __init__.py
│   └── ingest_pipeline.py          # Full end-to-end ingestion runner
│
├── tests/
│   ├── test_nlp.py
│   ├── test_graph.py
│   ├── test_vectorstore.py
│   └── test_api.py
│
├── notebooks/
│   └── exploration.ipynb           # EDA and pipeline prototyping
│
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions — test on push
│
├── docker-compose.yml              # Neo4j + Qdrant + API together
├── Dockerfile                      # FastAPI container
├── requirements.txt
├── .env.example
└── README.md
```

---

## Module-by-module breakdown

### `ingestion/` — pulls raw content from multiple sources

- **`file_loader.py`** — uses **PyMuPDF** (`fitz`) to extract text from PDFs;
  plain loaders for TXT/JSON.
- **`web_scraper.py`** — `requests` + `BeautifulSoup` to pull Wikipedia
  articles by topic (also handles simple news-API responses).
- **`cleaner.py`** — removes noise: HTML tags, extra whitespace, special
  characters, short meaningless sentences.

### `nlp/` — the brain of the pipeline

- **`ner.py`** — runs spaCy's `en_core_web_trf` model to extract entities
  (PERSON, ORG, GPE, PRODUCT, EVENT, CONCEPT).
- **`relation_extractor.py`** — uses the **Rebel** model from Hugging Face,
  which takes a sentence and outputs structured triples like
  `(Elon Musk, founded, SpaceX)`.
- **`coreference.py`** — resolves "he", "the company", "it" back to the
  actual named entity using **coreferee**.
- **`chunker.py`** — splits clean text into overlapping 512-token chunks
  (overlap 64) ready for embedding.

### `graph/` — everything Neo4j

- **`neo4j_client.py`** — manages the driver connection using env variables.
- **`graph_builder.py`** — takes NLP output (entities + relations) and
  converts them into Cypher **`MERGE`** statements, so re-ingest is idempotent
  — "SpaceX" stays a single node across 10 documents.
- **`graph_queries.py`** — reusable Cypher templates: find neighbours, find
  shortest path, get all relations of an entity.
- **`schema.py`** — node labels and relationship types as Python constants
  (single source of truth).

### `vectorstore/` — everything Qdrant

- **`embedder.py`** — loads `sentence-transformers/all-MiniLM-L6-v2` and
  produces **384-dimensional** embeddings for each text chunk.
- **`qdrant_client.py`** — handles connection and collection creation with
  **cosine** distance metric.
- **`indexer.py`** — batch-upserts all chunks into Qdrant with a rich
  payload: `source file, chunk index, entities mentioned, raw text`.

### `search/` — the query layer

- **`semantic_search.py`** — embeds the user's query and runs nearest-neighbour
  search in Qdrant, returning the top-*k* chunks with scores.
- **`graph_search.py`** — takes entity names and traverses Neo4j to find
  related entities, relationships, and paths.
- **`hybrid_search.py`** — **the key differentiator.** Runs semantic search
  first, extracts entity names from the top chunks, then enriches with graph
  context (neighbours + relationships).

### `api/` — FastAPI serving layer

Four routes:

- `POST /search` — natural-language query → ranked chunks + graph context.
- `GET /entity/{name}` — all relations and neighbours of a named entity.
- `GET /graph/{entity}` — subgraph (nodes + edges) around an entity, for
  visualisation.
- `GET /health` — uptime check.

### `pipeline/` — orchestration

- **`ingest_pipeline.py`** — the end-to-end runner: ingest → clean → NLP →
  write graph + vectors. Used by both CLI runs and CI smoke tests.

---

## Core libraries

| Purpose                  | Library                                         |
|--------------------------|-------------------------------------------------|
| NLP & NER                | `spacy`, `en_core_web_trf`                      |
| Relation extraction      | `transformers` (Rebel model from Hugging Face)  |
| Co-reference resolution  | `coreferee`                                     |
| Knowledge graph          | `neo4j` (Python driver)                         |
| Embeddings               | `sentence-transformers`                         |
| Vector store             | `qdrant-client`                                 |
| PDF ingestion            | `pymupdf` (`fitz`)                              |
| Web scraping             | `requests`, `beautifulsoup4`                    |
| API                      | `fastapi`, `uvicorn`, `pydantic`                |
| Text chunking            | `langchain` (text splitter only)                |
| Testing                  | `pytest`, `httpx`                               |

Full pinned list lives in [`requirements.txt`](./requirements.txt).

---

## Neo4j schema

```cypher
// Node types
(:Person       {name, source, mentions})
(:Organization {name, source, mentions})
(:Place        {name, source, mentions})
(:Concept      {name, source, mentions})
(:Document     {title, source, date})

// Relationship types
(:Person)-[:WORKS_AT]->(:Organization)
(:Person)-[:LOCATED_IN]->(:Place)
(:Person)-[:RELATED_TO]->(:Person)
(:Organization)-[:PART_OF]->(:Organization)
(:Concept)-[:RELATED_TO]->(:Concept)
(:Document)-[:MENTIONS]->(:Person|Organization|Place|Concept)
```

Property conventions:

- `name` is normalised lowercase for `MERGE` keys; the display form lives in
  a separate property when needed.
- `mentions` increments on each ingest so popular entities can be ranked by
  occurrence.
- `source` records the document title or URL of first appearance.

---

## API reference

### `POST /search`

Run a hybrid (vector + graph) query.

**Request**

```json
{
  "query": "Which companies has Elon Musk founded?",
  "top_k": 5,
  "include_graph": true
}
```

**Response**

```json
{
  "query": "Which companies has Elon Musk founded?",
  "chunks": [
    {
      "score": 0.87,
      "text": "Elon Musk founded SpaceX in 2002...",
      "source": "wikipedia/elon_musk",
      "entities": ["Elon Musk", "SpaceX"]
    }
  ],
  "graph": {
    "nodes": [{"id": "elon_musk", "label": "Person"}, ...],
    "edges": [{"from": "elon_musk", "to": "spacex", "type": "FOUNDED"}, ...]
  }
}
```

### `GET /entity/{name}`

Return all relations and neighbours of a named entity.

### `GET /graph/{entity}`

Return a subgraph (nodes + edges) around an entity for visualisation.

### `GET /health`

```json
{ "status": "ok", "neo4j": "up", "qdrant": "up" }
```

---

## Environment configuration

`.env.example` (copy to `.env` and fill in):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_chunks

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

---

## Docker setup

`docker-compose.yml` brings up Neo4j, Qdrant, and the API together:

```yaml
version: "3.8"
services:

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"   # Neo4j browser
      - "7687:7687"   # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/your_password
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - neo4j
      - qdrant

volumes:
  neo4j_data:
  qdrant_data:
```

- **Neo4j Browser** → http://localhost:7474
- **Qdrant Dashboard** → http://localhost:6333/dashboard
- **API docs (Swagger)** → http://localhost:8000/docs

---

## Step-by-step build plan

1. **Bring up the datastores.** `docker-compose up neo4j qdrant` — get both
   running locally before touching code.
2. **Build `ingestion/`.** Load a small Wikipedia sample (5–10 articles on a
   single topic, e.g. "AI companies") to test against.
3. **Build `nlp/`.** Run spaCy NER on the sample, print entities, verify they
   look right *before* wiring anything to a database.
4. **Build `graph/`.** Write the Cypher `MERGE` statements, open Neo4j Browser
   at http://localhost:7474, and visually confirm the graph is forming.
5. **Build `vectorstore/`.** Embed the chunks, index into Qdrant, run a manual
   similarity query.
6. **Build `search/hybrid_search.py`.** The core value-add — wire graph
   context into semantic results.
7. **Build `api/`.** Wrap everything in FastAPI endpoints.
8. **Tests + CI.** Fill out `tests/` and wire up `.github/workflows/ci.yml`.
9. **Demo dataset.** Add an AI-industry dataset and document sample queries in
   this README.

---

## Running locally

```bash
# 1. Clone & enter
cd knowledge-graph-semantic-search

# 2. Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m coreferee install en

# 3. Config
cp .env.example .env
# edit .env if your Neo4j password differs

# 4. Datastores
docker-compose up -d neo4j qdrant

# 5. Ingest the demo corpus
python -m pipeline.ingest_pipeline --source data/samples/

# 6. Serve the API
uvicorn api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for the live OpenAPI UI.

---

## Testing

```bash
pytest -q                 # full suite
pytest tests/test_nlp.py  # one module
pytest --cov=.            # with coverage
```

CI runs the same suite on every push via `.github/workflows/ci.yml`.

---

## Demo dataset & sample queries

The demo corpus lives in `data/samples/` — a small set of AI-industry
articles (OpenAI, Anthropic, DeepMind, etc.).

Try these once the pipeline is loaded:

```bash
# Hybrid search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Which AI labs work on alignment?", "top_k": 5}'

# Entity lookup
curl http://localhost:8000/entity/Anthropic

# Subgraph for visualisation
curl http://localhost:8000/graph/OpenAI
```

---

## Roadmap

- [ ] Multi-hop reasoning over the graph in `hybrid_search`.
- [ ] Streaming ingestion from RSS / news APIs.
- [ ] Switchable embedding backbones (BGE, E5, OpenAI).
- [ ] Front-end graph viewer (Cytoscape / D3) consuming `/graph/{entity}`.
- [ ] Re-ranking with a cross-encoder over the top-*k* chunks.

---

## License

TBD.
