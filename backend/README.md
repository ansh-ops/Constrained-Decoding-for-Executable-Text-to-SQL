# Backend API

Run locally:

```bash
cd "/Users/anshk/Desktop/deep learning project"
source venv/bin/activate
uvicorn backend.app:app --reload
```

Available routes:

- `GET /health`
- `GET /config`
- `POST /generate-sql`
- `POST /validate-sql`

Example request:

```json
{
  "question": "What is Terrence Ross' nationality?",
  "table_id": "1-10015132-16",
  "headers": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
  "mode": "fine_tuned",
  "temperature": 0.0
}
```
