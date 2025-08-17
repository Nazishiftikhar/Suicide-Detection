import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
import requests
import numpy as np
import smtplib
from groq import Groq
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

prediction = None

GROQ_API_KEY = "Enter your Groq API key here. Better to keep in .env file..."
client = Groq(api_key=GROQ_API_KEY)

if "chat_his" not in st.session_state:
    st.session_state.chat_his = [
        {
            "role": "system",
            "content":
                """You are a caring friend who listens and talks to someone going through a tough time. Respond in URDU, just like you would in a conversation.
                Your goal is to make the person feel heard, valued, and less alone. You're here to provide warmth, empathy, and gentle encouragement.
                When they talk to you, respond with a warm and caring statement. Ask open-ended questions that encourage them to share more about their feelings. Listen actively and acknowledge their emotions.

                For example, you could say:
                "Mujhe bohat khushi hui aap ne mujhse baat ki. Kya sab tumhe pareshaan kar raha hai?"
                "Aaj ka din kaafi mushkil lag raha hoga... Main yahan hoon sun’ne ke liye."

                Remember, be friendly, conversational, and non-judgmental. Validate their feelings and show empathy. Avoid giving advice or being too formal.
                Let's keep the conversation natural and supportive."""
        }
    ]


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------- Download model -------------------
@st.cache_data
def download_model():
    filename = "tfidf_vectorizer.pkl"
    return filename


# ----------------- Email Function -------------------
def send_email_alert(subject, body, to_emails, from_email=None):
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        sender_password = st.secrets["EMAIL_PASSWORD"]
    except Exception:
        st.error("Email credentials not found. Set EMAIL_ADDRESS and EMAIL_PASSWORD in Streamlit secrets.")
        return

    msg = MIMEMultipart()
    msg['From'] = from_email if from_email else sender_email
    msg['To'] = ", ".join(to_emails)
    msg['Reply-To'] = from_email if from_email else sender_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_emails, msg.as_string())
        st.info("🚨 Alert email sent successfully.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")


def ai(user_query: str):
    # Add the new user message to history
    st.session_state.chat_his.append({"role": "user", "content": user_query})

    # Generate AI response
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=st.session_state.chat_his
    )
    model_reply = response.choices[0].message.content.strip()

    # Add assistant reply to history
    st.session_state.chat_his.append({"role": "assistant", "content": model_reply})

    return model_reply


def ai_with_history():
    """Send entire history without duplicating messages."""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=st.session_state.chat_his
    )
    model_reply = response.choices[0].message.content.strip()
    return model_reply


# ----------------- Load model & vectorizer -------------------
@st.cache_resource
def load_model():

    model_path = download_model()
    return joblib.load(model_path)


model = load_model()


@st.cache_resource
def load_vectorizer():

    if not os.path.exists("tfidf_vectorizer.pkl"):
        st.error("TF-IDF vectorizer file not found. Please upload it.")
        st.stop()
    return joblib.load("tfidf_vectorizer.pkl")


vectorizer = load_vectorizer()

# ----------------- Top Navigation Menu -------------------
selected = option_menu(
    menu_title=None,  # No title
    options=["Home", "Sentiment Analysis", "About"],
    icons=["house", "emoji-smile", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)
# ----------------- Home Page -------------------
if selected == "Home":

    # Main title with style
    st.markdown("""
        <h1 style='text-align: center; color: #6c63ff; font-size: 40px;'>
            🏠 Welcome to the Suicidal Thought Detection System
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size: 18px; color: #555;'>
            Your mental health matters. This system analyzes your text to detect early signs of suicidal thoughts and provide timely alerts.
        </p>
        <br>
    """, unsafe_allow_html=True)

# ----------------- Sentiment Analysis Page -------------------
elif selected == "Sentiment Analysis":
    st.title("🧠 Suicidal Ideation Detection")

    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

    # User enters their email
    user_email = st.text_input("Enter your email (optional):")

    # First input
    text_input = st.text_area("Enter text to analyze sentiment:")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            X_input = vectorizer.transform([text_input])
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Sentiment: **{prediction}**")

        if prediction == "Suicidal":
            st.error("⚠️ Suicide sentiment detected. Alert triggered.")
            # Alert
            send_email_alert(
                subject="🚨 Suicide Sentiment Detected",
                body=f"The following message indicates suicidal intent:\n\n{text_input}",
                to_emails=["nazishiftikhar112@gmail.com"],
                from_email=user_email
            )

            # Show chat section
            st.session_state.show_chat = True

            # First therapeutic model reply
            reply = ai(text_input)

            st.session_state.chat_his = [
                st.session_state.chat_his[0],  # system prompt
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": reply}
            ]

        else:
            st.session_state.show_chat = False
            st.info("💡 Have a good day.")

    # Chat expander
    if st.session_state.show_chat:
        with st.expander("💬 Support Chat", expanded=True):
            # Display history
            for msgs in st.session_state.chat_his:
                if msgs["role"] == "system":
                    continue
                st.chat_message(msgs["role"]).write(msgs["content"])

            # Input box for continuation
            if prompt := st.chat_input("Type your message..."):
                st.session_state.chat_his.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                reply = ai_with_history()
                st.session_state.chat_his.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)

# ----------------- About Page -------------------
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This web application analyzes the **sentiment** of user-provided text.

    - 🧠 Trained with real-world data  
    - 🛡️ Sends email alerts if **suicidal intent** is detected  
    - 🚀 Simple and fast sentiment classification  

    👉 Navigate to the **Sentiment Analysis** tab above to begin.

    **Developer**: Nazish Iftikhar  
    **Model Hosted On**: [Hugging Face](https://huggingface.co/naziiiii/Sentiments/blob/main/voting_model.pkl)  
    📫 **Contact**: `nazivirk113@gmail.com`
    """)
