# ==========================================================================
# COMMENT ANALYSIS DASHBOARD - Final Premium UX/UI + IA Chat Restored
# Updated:
# - Sentiment donut chart without center comment count
# - Top emojis as HTML/CSS horizontal bars
# ==========================================================================

import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from openai import OpenAI
import emoji
from collections import Counter
import concurrent.futures
import json
from matplotlib.colors import ListedColormap

# ----------------------------- PAGE CONFIG --------------------------------
st.set_page_config(page_title="Comment Analysis Dashboard", layout="wide")

# ------------------------------ STYLES -------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* GLOBAL */
.stApp {
  background-color: #FAFAFA;
  font-family: 'Inter', sans-serif;
}

/* CARDS */
[data-testid="stHorizontalBlock"] > div {
  background-color:#FFFFFF;
  border-radius:18px;
  padding:24px;
  box-shadow:0 4px 10px rgba(0,0,0,0.05);
  margin-bottom:20px;
}

/* HEADERS */
h1,h2,h3,h4 {
  color:#111111;
  font-weight:600;
}

/* BUTTONS */
.stButton>button {
  background:linear-gradient(90deg,#8A2BE2 0%,#6C63FF 100%);
  color:#fff;
  border:none;
  border-radius:10px;
  padding:10px 24px;
  font-weight:600;
  transition:.2s;
}

.stButton>button:hover {
  transform:scale(1.02);
  background:linear-gradient(90deg,#7B1FA2 0%,#5A55FF 100%);
}

/* GENERIC CONTAINERS */
.st-emotion-cache-1r4qj8v,
.st-emotion-cache-0 {
  background-color:#FFFFFF;
  border-radius:18px;
  padding:24px;
  box-shadow:0 4px 12px rgba(0,0,0,0.05);
}

/* CHAT IA */
.chat-card {
  background:#fff;
  border:1px solid #eee;
  border-radius:14px;
  padding:14px 16px;
  margin:8px 0;
}

.role-user {
  background:#F5F7FF;
}

.role-assistant {
  background:#F9F9F9;
}

/* ------------------- EMOJI HORIZONTAL BARS ------------------- */
.emoji-bars-wrapper {
  margin-top:8px;
}

.emoji-bar-row {
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom:18px;
}

.emoji-bar-emoji {
  width:44px;
  min-width:44px;
  text-align:center;
  font-size:2rem;
  line-height:1;
  font-family:
    "Apple Color Emoji",
    "Segoe UI Emoji",
    "Noto Color Emoji",
    "Twemoji Mozilla",
    sans-serif;
}

.emoji-bar-track {
  flex:1;
  height:18px;
  background:#F1F1F5;
  border-radius:999px;
  overflow:hidden;
  position:relative;
}

.emoji-bar-fill {
  height:100%;
  border-radius:999px;
}

.emoji-bar-badge {
  min-width:58px;
  text-align:center;
  color:#fff;
  font-weight:700;
  font-size:.95rem;
  padding:6px 12px;
  border-radius:10px;
}

/* Shared color classes */
.pink {
  background:linear-gradient(90deg,#E11D74 0%,#F72585 100%);
}

.orange {
  background:linear-gradient(90deg,#FF6B00 0%,#FF8A3D 100%);
}

.yellow {
  background:linear-gradient(90deg,#F4B400 0%,#FFC83D 100%);
}

.purple {
  background:linear-gradient(90deg,#8A2BE2 0%,#A855F7 100%);
}

.red {
  background:linear-gradient(90deg,#E91E63 0%,#FF4D8D 100%);
}
</style>
""", unsafe_allow_html=True)

# --------------------------- CORE FUNCTIONS --------------------------------
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def parse_ai_batch_response(response_text, original_batch):
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        else:
            json_str = response_text.strip()

        analyses = json.loads(json_str)

        if isinstance(analyses, dict) and "analyses" in analyses:
            analyses = analyses["analyses"]

        if not isinstance(analyses, list):
            raise ValueError("AI response was not a JSON array.")

        for i, analysis in enumerate(analyses):
            if i < len(original_batch):
                analysis["Original Comment"] = original_batch[i]

        return analyses

    except Exception:
        return [
            {
                "Original Comment": comment,
                "Sentiment": "Neutral",
                "Explanation": "Unanalyzable content."
            }
            for comment in original_batch
        ]


@st.cache_data(ttl="24h")
def analyze_comment_batch_cached(_api_key, model_name, comment_batch):
    client = OpenAI(api_key=_api_key)

    comments_str = "\n".join(
        [f'{i + 1}. "{comment}"' for i, comment in enumerate(comment_batch)]
    )

    instructions = """
    You are a sentiment analysis API for customer and social media comments.
    Return only valid JSON. Do not wrap the response in Markdown.
    For each input comment, return exactly one object with:
    - Sentiment: Positive, Negative, or Neutral
    - Explanation: one concise sentence explaining the classification
    """

    prompt = f"""
    Analyze each numbered comment below.

    Return a JSON array in the same order as the comments:
    [{{"Sentiment":"Positive|Negative|Neutral", "Explanation":"..."}}]

    Comments:
    {comments_str}
    """

    try:
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=prompt
        )
        return parse_ai_batch_response(response.output_text, comment_batch)

    except Exception:
        return [
            {
                "Original Comment": comment,
                "Sentiment": "Neutral",
                "Explanation": "API error."
            }
            for comment in comment_batch
        ]


def run_batch_analysis(api_key, model_name, comments):
    results = []
    batch_size = 50

    batches = [
        comments[i:i + batch_size]
        for i in range(0, len(comments), batch_size)
    ]

    with st.spinner(f"Analyzing {len(comments)} comments..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    analyze_comment_batch_cached,
                    api_key,
                    model_name,
                    batch
                )
                for batch in batches
            ]

            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

    return pd.DataFrame(results)


def load_comments(uploaded_file, gsheets_link, text_input):
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    if gsheets_link:
        url_csv = (
            gsheets_link
            .replace("/edit?usp=sharing", "/export?format=csv")
            .split("/edit")[0]
            + "/export?format=csv"
        )

        df = pd.read_csv(
            url_csv,
            header=None,
            engine="python",
            on_bad_lines="skip"
        )

        return df.iloc[:, 0].dropna().astype(str).tolist()

    if text_input:
        return [
            line.strip()
            for line in text_input.split("\n")
            if line.strip()
        ]

    return []


def generate_visuals(df):
    visuals = {}

    if df.empty:
        return visuals

    # -------------------- SENTIMENT DONUT CHART --------------------
    df["Sentiment"] = df["Sentiment"].str.strip()
    sentiment_counts = df["Sentiment"].value_counts()

    sentiment_color_map = {
        "Positive": "#2ecc71",
        "Neutral": "#f1c40f",
        "Negative": "#e74c3c"
    }

    sentiment_colors = [
        sentiment_color_map.get(sentiment, "#cccccc")
        for sentiment in sentiment_counts.index
    ]

    fig_sentiment, ax_sentiment = plt.subplots(figsize=(5, 5))
    fig_sentiment.patch.set_alpha(0)
    ax_sentiment.set_facecolor("none")

    wedges, texts, autotexts = ax_sentiment.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        colors=sentiment_colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.0f}%",
        pctdistance=0.78,
        textprops={
            "fontsize": 11,
            "color": "#111111"
        },
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white",
            "linewidth": 3
        }
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("600")
        autotext.set_color("#111111")

    ax_sentiment.set_title(
        "Sentiment Breakdown",
        fontsize=14,
        fontweight="600",
        pad=16
    )

    ax_sentiment.axis("equal")

    visuals["sentiment_chart"] = fig_sentiment

    # ------------------------- WORD CLOUD --------------------------
    text = " ".join(df["Original Comment"].dropna())

    stopwords_es = set(
        list(STOPWORDS) + [
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
            "las", "por", "un", "para", "con", "no", "una", "su", "al",
            "lo", "como", "más", "sus", "le", "ya", "o", "este", "ha",
            "me", "si", "mi", "yo", "porque", "esta", "muy", "sin",
            "sobre", "también", "fue", "esa", "son", "está", "ni",
            "solo", "puede", "uno", "delos"
        ]
    )

    colorful_map = ListedColormap(
        [
            "#FF5722",
            "#4CAF50",
            "#2196F3",
            "#FFC107",
            "#9C27B0",
            "#E91E63",
            "#00BCD4"
        ]
    )

    word_cloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=colorful_map,
        prefer_horizontal=0.9,
        collocations=False,
        stopwords=stopwords_es,
        max_words=60,
        max_font_size=110,
        min_font_size=15,
        margin=10,
        relative_scaling=0.3
    ).generate(text)

    fig_word_cloud, ax_word_cloud = plt.subplots()
    ax_word_cloud.imshow(word_cloud, interpolation="bilinear")
    ax_word_cloud.axis("off")

    visuals["word_cloud"] = fig_word_cloud

    # -------------------- TOP EMOJIS DATA -------------------------
    all_emojis = [
        character
        for character in "".join(df["Original Comment"].dropna())
        if character in emoji.EMOJI_DATA
    ]

    if all_emojis:
        visuals["emoji_ranking"] = Counter(all_emojis).most_common(5)

    return visuals


def render_emoji_bars(emoji_ranking):
    if not emoji_ranking:
        st.info("No emojis found in the analyzed comments.")
        return

    color_classes = ["pink", "orange", "yellow", "purple", "red"]
    max_count = max(count for _, count in emoji_ranking)

    st.subheader("Top Emojis")
    st.markdown('<div class="emoji-bars-wrapper">', unsafe_allow_html=True)

    for idx, (emoji_char, count) in enumerate(emoji_ranking):
        color_class = color_classes[idx % len(color_classes)]

        percentage_width = (count / max_count) * 100 if max_count > 0 else 0
        percentage_width = max(percentage_width, 12)

        st.markdown(
            f"""
            <div class="emoji-bar-row">
                <div class="emoji-bar-emoji">{emoji_char}</div>
                <div class="emoji-bar-track">
                    <div class="emoji-bar-fill {color_class}" style="width:{percentage_width}%;"></div>
                </div>
                <div class="emoji-bar-badge {color_class}">x{count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------- STATE ---------------------------------------
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------ HEADER -------------------------------------
st.title("Sentiment Analysis Dashboard")

# API key
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    openai_model = st.secrets.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

except Exception:
    api_key = ""
    openai_model = DEFAULT_OPENAI_MODEL

if not api_key:
    st.info(
        "Add your OpenAI API key to continue. "
        "This key is used only for this Streamlit session and is not saved here."
    )
    api_key = st.text_input("OpenAI API key", type="password")

if not api_key:
    st.stop()


# ------------------------------- INPUT -------------------------------------
if st.session_state.analysis_df is None:
    st.header("Load Comments")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    gsheets_link = st.text_input("Google Sheets Link")
    text_input = st.text_area("Or paste comments manually:", height=200)

    if st.button("✨ Analyze Now!"):
        comments = load_comments(uploaded_file, gsheets_link, text_input)

        if comments:
            st.session_state.analysis_df = run_batch_analysis(
                api_key,
                openai_model,
                comments
            )
            st.rerun()

        else:
            st.warning("Please provide comments to analyze.")

else:
    df = st.session_state.analysis_df
    visuals = generate_visuals(df)

    # -------------------------- IA CHAT --------------------------
    with st.expander("💬 IA Chat sobre estos resultados", expanded=False):

        for message in st.session_state.chat_history:
            role_class = (
                "role-user"
                if message["role"] == "user"
                else "role-assistant"
            )

            with st.container():
                st.markdown(
                    f"""
                    <div class="chat-card {role_class}">
                        <b>{message["role"].title()}:</b><br>
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        user_question = st.chat_input("Haz una pregunta sobre el análisis...")

        if user_question:
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            sentiment_counts = df["Sentiment"].value_counts().to_dict()
            sample_rows = df.head(30).to_string(index=False)

            context = f"""
            Sentiment counts: {sentiment_counts}

            Sample rows up to 30:
            {sample_rows}
            """

            try:
                client = OpenAI(api_key=api_key)
                response = client.responses.create(
                    model=openai_model,
                    instructions=(
                        "You are an expert business analyst. "
                        "Answer concisely using only the given data."
                    ),
                    input=(
                        f"--- DATA ---\n{context}\n--- END DATA ---\n"
                        f"QUESTION: {user_question}"
                    )
                )

                answer = response.output_text.strip()

            except Exception as error:
                answer = f"No pude completar la respuesta ({error})."

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()

    # -------------------------- MAIN GRID --------------------------
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        if "sentiment_chart" in visuals:
            with st.container():
                st.pyplot(visuals["sentiment_chart"])

    with col2:
        with st.container():
            if "emoji_ranking" in visuals:
                render_emoji_bars(visuals["emoji_ranking"])
            else:
                st.subheader("Top Emojis")
                st.info("No emojis found in the analyzed comments.")

    if "word_cloud" in visuals:
        with st.container():
            st.subheader("Word Cloud")
            st.pyplot(visuals["word_cloud"])

    with st.container():
        st.subheader("Detailed Data")
        st.dataframe(df)
