# Sentiment Analysis Dashboard

App Streamlit para analizar sentimientos de comentarios usando la API de OpenAI.

## Archivos que debes subir a GitHub

- `app.py`
- `requirements.txt`

No subas ninguna API key a GitHub.

## En Streamlit Cloud

Cuando crees la app:

- Repository: tu repositorio de GitHub
- Branch: `main`
- Main file path: `app.py`

En Settings > Secrets, pega:

```toml
OPENAI_API_KEY = "sk-proj-tu-clave-real"
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_CHAT_MODEL = "gpt-5.5"
```

`OPENAI_MODEL` se usa para clasificar muchos comentarios.
`OPENAI_CHAT_MODEL` se usa para el chat estrategico de la app. Si `gpt-5.5` no esta disponible en tu cuenta, la app intentara responder con `OPENAI_MODEL`.

Si no pones la clave en Secrets, la app te la pedira en pantalla como campo de contrasena.
