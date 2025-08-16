import streamlit as st
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
import praw
import json
from pathlib import Path
import google.generativeai as genai
import random
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import base64
import tempfile
from xhtml2pdf import pisa

genai.configure(api_key=st.secrets["google"]["gemini_api_key"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Emotion Classifier ==========
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=-1
)

# ========== Reddit API Setup ==========
reddit = praw.Reddit(
    client_id=st.secrets["reddit"]["client_id"],
    client_secret=st.secrets["reddit"]["client_secret"],
    user_agent=st.secrets["reddit"]["user_agent"]
)

if "showed_welcome_block" not in st.session_state:
    st.session_state.showed_welcome_block = False


# ========== Memory Utilities ==========
def save_conversation(username, history):
    with open(f"{username}_history.json", "w") as f:
        json.dump(history, f)

def load_conversation(username):
    file = Path(f"{username}_history.json")
    if file.exists():
        with open(file, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(i, (list, tuple)) and len(i) == 2 for i in data):
            return [{"user": i[0], "bot": i[1]} for i in data]
        return data
    return []

def generate_psychologist_summary(chat_history, reddit_context, personality_summary):
    prompt = (
        "You are a clinical psychologist preparing for a first session with a new client. "
        "Based on their Reddit activity, inferred personality traits, and prior chatbot conversations, "
        "write a 1-page briefing that summarizes the user's emotional tendencies, recurring concerns, and communication patterns.\n\n"
        f"Reddit context:\n{reddit_context}\n\n"
        f"Personality summary:\n{personality_summary}\n\n"
        f"Chat history (last 10 messages):\n" +
        "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in chat_history[-10:]]) +
        "\n\nGenerate the summary in a clinical but readable tone, suitable for another therapist."
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Summary could not be generated."
    except Exception as e:
        print("Gemini summary error:", str(e))
        return "Error generating summary."



# ========== Emotion ==========
def get_emotion_scores(text, min_confidence=0.4, fallback_label="neutral"):
    try:
        raw_scores = classifier(text)[0]
        sorted_emotions = sorted(raw_scores, key=lambda x: x['score'], reverse=True)

        # Fallback if no emotion is above threshold
        if sorted_emotions[0]['score'] < min_confidence:
            dominant_emotion = fallback_label
        else:
            dominant_emotion = sorted_emotions[0]['label']

        return sorted_emotions, dominant_emotion

    except Exception as e:
        print(f"[Emotion Classifier Error] {e}")
        return [], fallback_label

def summarize_user_personality(reddit_context):
    prompt = (
        "You are a psychologist analyzing a user's Reddit activity to extract key personality traits, interests, and recurring emotional patterns. "
        "Summarize this into a psychological profile. Be clinical but concise.\n\n"
        f"Reddit context:\n{reddit_context}\n\nPersonality summary:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Personality traits could not be derived."
    except Exception:
        return "Personality traits could not be derived."



def get_user_reddit_context(username, max_items=100):
    try:
        user = reddit.redditor(username)
        comments = [
            f"Comment in r/{c.subreddit.display_name}: {c.body.strip()}"
            for c in user.comments.new(limit=max_items) if len(c.body.strip()) > 20
        ]
        submissions = [
            f"Post in r/{s.subreddit.display_name}: {s.title.strip()} - {s.selftext.strip()}"
            for s in user.submissions.new(limit=max_items) if len(s.title.strip()) > 10
        ]

        combined = comments + submissions
        most_recent = combined[:20]
        older = combined[20:]
        random.shuffle(older)
        selected = most_recent + older[:30]  # 20 recent and 30 random older

        subreddits = {c.subreddit.display_name for c in user.comments.new(limit=max_items)}
        subreddits.update({s.subreddit.display_name for s in user.submissions.new(limit=max_items)})

        summary = f"User is active in: {', '.join(subreddits)}.\n"
        summary += "\n".join(selected[:12])  # limit for readability
        return summary.strip()
    except Exception as e:
        print("Reddit fetch error:", str(e))
        return "Could not retrieve Reddit context."

# ========== Prompt Builder ==========
# This was out first attempt, but then got replaced by gemini
def build_prompt_with_history(user_input, emotion_scores, reddit_context="", history=None, max_turns=3):
    dominant_emotion = emotion_scores[0]['label']
    sensitive_emotions = {e['label']: e['score'] for e in emotion_scores if e['label'] in ["sadness", "anger", "fear"]}
    needs_care = any(score > 0.01 for score in sensitive_emotions.values())

    context_intro = f"This is the user's Reddit context: {reddit_context} " if reddit_context else ""
    emotion_notice = f"The user may be feeling distressed ({', '.join(sensitive_emotions.keys())}). Respond in a calm and supportive way. " if needs_care else ""

    history_text = ""
    if history:
        history_text = "Conversation history:\n" + "\n".join([f"User: {u['user']}\nBot: {u['bot']}" for u in history[-max_turns:]]) + "\n"

    prompt = f"{context_intro}{emotion_notice}{history_text}User: {user_input}"
    return " ".join(prompt.split()[:120]) + "..."

# ========== Generate Chatbot Response ==========
def generate_gemini_reply(user_input, emotion_scores, reddit_context="", history=None):

    if not emotion_scores:
        dominant_emotion = "neutral"
        dominant_score = 0.0
        emotion_scores = [{"label": "neutral", "score": 0.0}]
    else:
        dominant_emotion = emotion_scores[0]['label']
        dominant_score = emotion_scores[0]['score']

    sensitive_emotions = {
        e['label']: e['score']
        for e in emotion_scores
        if e['label'] in ["sadness", "anger", "fear"]
    }
    needs_care = any(score > 0.01 for score in sensitive_emotions.values())

    # Efficient and behaviorally consistent prompt
    if not st.session_state.get("base_instruction_shown", False):
        prompt_parts = [
            "You are a supportive AI psychologist with insight into the user's personality.",
            "Be empathetic, concise, and psychologically informed. Keep responses under 10 lines. Remember this.",
            "Remember that you are psychologist, so you can help the users with their emotions. This means, that you can also just listen and don't directly suggest what to do.",
            "If the user is not asking for a suggestion what to do then don't give one!"
        ]
        st.session_state.base_instruction_shown = True
    else:
        prompt_parts = []


    if needs_care:
        prompt_parts.append("The user may be experiencing emotional distress. Be especially gentle and supportive.")

    # Only include personality summary once
    if not st.session_state.get("first_prompt_done", False):
        if personality_summary := st.session_state.get("personality_summary"):
            prompt_parts.append(f"User personality profile: {personality_summary}. Use this personality throughout the whole conversation and reply accordingly.")
        st.session_state.first_prompt_done = True 

    personality_summary = st.session_state.get("personality_summary")

    # Only include history summary once
    if history and not st.session_state.get("history_context_shown", False):
        recent_dialogue = "\n".join([
            f"User: {h['user']}\nBot: {h['bot']}"
            for h in history[-3:]
        ])
        prompt_parts.append("Previous conversation summary:\n" + recent_dialogue)
        st.session_state.history_context_shown = True

    reddit_context = st.session_state.get("reddit_context", "")

    prompt_parts.append(
        f"Based on the personality profile ({personality_summary}) and emotional state({dominant_emotion}), suggest actionable activities if the user asks what to do or you feel a suggestion might be helpful."
        f"Keep in mind that you don't always need to suggest someting. If the user only tells you about a specific bad thing that happened, just reply as a normal human without specific suggestion."
        f"Suggest something only if you think that the user is asking for recommendations."
        f"Prioritize actions aligned with the user's interests and traits. So definitely include the personality and thus the reddit content! This is the reddit context {reddit_context}"
        f"Keep the tone supportive and warm, and the suggestion under 10 lines."
        )

    prompt_parts.append(f"User: {user_input}\nBot:")

    full_prompt = "\n\n".join(prompt_parts)

    try:
        response = gemini_model.generate_content(full_prompt)
        if not response or not response.text.strip():
            fallback = "I'm here for you. Can you tell me a bit more so I can help better?"
            return dominant_emotion, dominant_score, fallback
        return dominant_emotion, dominant_score, response.text.strip()
    except Exception as e:
        print("Gemini error:", str(e))
        return dominant_emotion, dominant_score, "I'm having trouble responding right now. Please try again in a moment."


# ========== Streamlit App ==========
st.set_page_config(page_title=" Mental Health Chatbot", page_icon="💬", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Sidebar Summary Button ==========
with st.sidebar:
    st.markdown("### Tools")
    reddit_ready = "reddit_context" in st.session_state and st.session_state.reddit_context.strip()
    generate_disabled = not reddit_ready
    if st.button("📝 Generate Psychologist Summary", disabled=generate_disabled):
        if reddit_ready:
            summary_text = generate_psychologist_summary(
                st.session_state.get("history", []),
                st.session_state.get("reddit_context", ""),
                st.session_state.get("personality_summary", "")
            )

            html_ready_summary = summary_text.replace('\n', '<br>')
            styled_html = f"""
            <html>
            <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    text-align: center;
                    color: #4a90e2;
                }}
                h2 {{
                    margin-top: 2em;
                    color: #444;
                }}
                p {{
                    margin: 1em 0;
                }}
            </style>
            </head>
            <body>
            <h1>Psychologist Summary Report</h1>
            <h2>Client: {st.session_state.username}</h2>
            <p>{html_ready_summary}</p>
            </body>
            </html>
            """

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pisa.CreatePDF(styled_html, dest=tmp_pdf)
                tmp_pdf.seek(0)
                st.download_button(
                label="📥 Download Psychologist PDF",
                data=tmp_pdf.read(),
                file_name=f"{st.session_state.username}_psychologist_summary.pdf",
                mime="application/pdf"
                )

# ========== Initial User Setup ==========
if "username" not in st.session_state:
    with stylable_container(
        key="login",
        css_styles="""
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            margin-top: 2rem;
            text-align: center;
        """,
    ):
        st.image("assets/joao_100x100.png", width=100)
        st.markdown("<h2 style='margin-top: 1rem;'>👋 Meet <span style='color:#4a90e2'>Joao</span></h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: #555;'>Your empathetic mental health companion. Let's chat about how you're feeling today—no judgment, just support.</p>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        username_input = st.text_input("🔍 Enter your Reddit username", placeholder="e.g. kind_redditor23")

        st.markdown("")  # spacer

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if st.button("💬 Start Chatting with Joao", use_container_width=True):
                if username_input.strip():
                    with st.spinner("Retrieving Reddit data and preparing your session..."):
                        st.session_state.username = username_input.strip()
                        loaded_history = load_conversation(username_input)

                        st.session_state.history = loaded_history
                        st.session_state.is_new_user = not bool(loaded_history)
                        st.session_state.just_resumed = bool(loaded_history)

                        if "showed_welcome_block" not in st.session_state:
                            st.session_state.showed_welcome_block = False
                        if bool(loaded_history):
                            st.session_state.show_welcome_back_now = True

                        st.session_state.reddit_context = get_user_reddit_context(username_input)
                        st.session_state.personality_summary = summarize_user_personality(st.session_state.reddit_context)
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a valid Reddit username.")
    st.stop()

# ========== Welcome Joao after Login ==========
if "username" in st.session_state:
    if "is_new_user" in st.session_state:
        del st.session_state["is_new_user"]
    with stylable_container(
        key="intro_after_login",
        css_styles="""
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            margin-top: 2rem;
            text-align: center;
        """,
    ):
        st.image("assets/joao_100x100.png", width=100)  # Re-display Joao image
        st.markdown("<h2 style='margin-top: 1rem;'>Welcome, <span style='color:#4a90e2'>Joao</span> is here for you</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: #555;'>Feel free to tell Joao how you're feeling. "
            "He's here to listen, support, and chat at your pace — no pressure.</p>",
            unsafe_allow_html=True
        )

# ========== Chat Interface ==========

if st.session_state.get("just_resumed"):
    with stylable_container(
        key="history_info",
        css_styles="""
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: #444;
        """,
    ):
        st.markdown("🕘 **These messages are from your last session. Let's pick up where we left off or start fresh.**")
    st.session_state.just_resumed = False

# Display chat history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["user"])
    with st.chat_message("assistant"):
        st.markdown(entry["bot"])

# Display Joao and welcome message at the top of chat interface
if st.session_state.get("show_welcome_back_now", False) and not st.session_state.get("showed_welcome_block", False):
    with stylable_container(
        key="welcome_back",
        css_styles="""
            background-color: #f4f4f5;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 2rem;
        """
    ):
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("assets/joao_100x100.png", width=80)
        with col2:
            st.markdown("<h2 style='margin-bottom: 0;'>Welcome back, <span style='color:#4a90e2'>Joao</span> is here for you</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #666;'>Feel free to tell Joao how you're feeling. He's here to listen, support, and chat at your pace — no pressure.</p>", unsafe_allow_html=True)
            st.info("⏰ These messages are from your last session. Let's pick up where we left off or start fresh.")
    st.session_state.showed_welcome_block = True
    st.session_state.show_welcome_back_now = False

# Handle chat input
user_input = st.chat_input("How are you feeling today?")
if user_input:
    # Get sorted emotion scores and dominant emotion
    emotion_scores, dominant_emotion = get_emotion_scores(user_input)
    confidence = emotion_scores[0]["score"] if emotion_scores else 0.0

    # Generate Gemini response
    _, _, response = generate_gemini_reply(
        user_input,
        emotion_scores,
        st.session_state.reddit_context,
        st.session_state.history,
    )

    # Display conversation
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save to session history
    st.session_state.history.append({"user": user_input, "bot": response})
    save_conversation(st.session_state.username, st.session_state.history)
