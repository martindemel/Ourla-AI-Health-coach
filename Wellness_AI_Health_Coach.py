import os
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
import openai
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import calendar
import math

# Try importing statsmodels for trendline; if not available, omit trendline.
try:
    import statsmodels.api as sm
    trendline_option = "ols"
except ImportError:
    trendline_option = None

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()  # Loads variables from .env file
OURA_PAT = os.getenv("OURA_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)  # Instantiate client per migration guide.
headers = {"Authorization": f"Bearer {OURA_PAT}"}

# --------------------------
# Date Configuration
# --------------------------
today = datetime.today().date()
objective_date = today - timedelta(days=1)  # For objective data, we use yesterday
today_iso = today.strftime("%Y-%m-%d")
today_us = today.strftime("%m/%d/%Y")
objective_date_str = objective_date.strftime("%Y-%m-%d")
objective_date_us = objective_date.strftime("%m/%d/%Y")

# --------------------------
# Database File
# --------------------------
db_path = "oura_sleep_coach.db"

# --------------------------
# Database Initialization & Functions
# --------------------------
def init_db():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sleep (
            date TEXT PRIMARY KEY,
            sleep_score INTEGER,
            total_sleep_minutes INTEGER,
            deep_sleep_minutes INTEGER,
            rem_sleep_minutes INTEGER,
            sleep_efficiency INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS readiness (
            date TEXT PRIMARY KEY, 
            readiness_score INTEGER, 
            hrv_balance INTEGER, 
            resting_hr INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            date TEXT PRIMARY KEY, 
            activity_score INTEGER, 
            steps INTEGER, 
            total_calories INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spo2 (
            date TEXT PRIMARY KEY, 
            avg_spo2 REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_metrics (
            date TEXT PRIMARY KEY,
            readiness_rating INTEGER,
            mood INTEGER,
            energy INTEGER,
            stress INTEGER,
            soreness INTEGER
        )
    """)
    conn.commit()
    conn.close()

def fetch_oura_data(url, params, table_name, insert_sql, extract_func):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    resp = requests.get(url, headers=headers, params=params)
    if resp.ok:
        data = resp.json().get("data", [])
        for record in data:
            values = extract_func(record)
            cur.execute(insert_sql, values)
        conn.commit()
        conn.close()
    else:
        conn.close()
        raise Exception(f"Error fetching {table_name} data: {resp.status_code} {resp.text}")

def extract_sleep(record):
    date = record.get("day")
    score = record.get("score")
    contrib = record.get("contributors", {})
    total_sleep = contrib.get("total_sleep")
    deep_sleep = contrib.get("deep_sleep")
    rem_sleep = contrib.get("rem_sleep")
    efficiency = contrib.get("efficiency")
    return (date, score, total_sleep, deep_sleep, rem_sleep, efficiency)

def extract_readiness(record):
    date = record.get("day")
    score = record.get("score")
    contrib = record.get("contributors", {})
    hrv_balance = contrib.get("hrv_balance")
    resting_hr = contrib.get("resting_heart_rate")
    return (date, score, hrv_balance, resting_hr)

def extract_activity(record):
    date = record.get("day")
    score = record.get("score")
    steps = record.get("steps")
    calories = record.get("total_calories")
    return (date, score, steps, calories)

def extract_spo2(record):
    date = record.get("day")
    spo2 = (record.get("spo2_percentage") or {}).get("average")
    return (date, spo2)

# --------------------------
# Update Objective Data & Set Last Updated Timestamp
# --------------------------
def update_oura_data():
    init_db()  # Ensure tables exist
    cutoff = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    start = cutoff
    fetch_oura_data(
        url="https://api.ouraring.com/v2/usercollection/daily_sleep",
        params={"start_date": start, "end_date": today_iso},
        table_name="sleep",
        insert_sql="""
            INSERT OR REPLACE INTO sleep 
            (date, sleep_score, total_sleep_minutes, deep_sleep_minutes, rem_sleep_minutes, sleep_efficiency)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
        extract_func=extract_sleep
    )
    fetch_oura_data(
        url="https://api.ouraring.com/v2/usercollection/daily_readiness",
        params={"start_date": start, "end_date": today_iso},
        table_name="readiness",
        insert_sql="""
            INSERT OR REPLACE INTO readiness (date, readiness_score, hrv_balance, resting_hr)
            VALUES (?, ?, ?, ?)
        """,
        extract_func=extract_readiness
    )
    fetch_oura_data(
        url="https://api.ouraring.com/v2/usercollection/daily_activity",
        params={"start_date": start, "end_date": today_iso},
        table_name="activity",
        insert_sql="""
            INSERT OR REPLACE INTO activity (date, activity_score, steps, total_calories)
            VALUES (?, ?, ?, ?)
        """,
        extract_func=extract_activity
    )
    fetch_oura_data(
        url="https://api.ouraring.com/v2/usercollection/daily_spo2",
        params={"start_date": start, "end_date": today_iso},
        table_name="spo2",
        insert_sql="""
            INSERT OR REPLACE INTO spo2 (date, avg_spo2)
            VALUES (?, ?)
        """,
        extract_func=extract_spo2
    )
    st.session_state.last_updated = datetime.now().strftime("%m/%d/%Y %I:%M %p")

# --------------------------
# Load Data Function
# --------------------------
def load_data():
    conn = sqlite3.connect(db_path)
    query = """
        SELECT s.date, s.sleep_score, s.total_sleep_minutes, s.deep_sleep_minutes, s.rem_sleep_minutes, s.sleep_efficiency,
               r.readiness_score, r.hrv_balance, r.resting_hr,
               a.activity_score, a.steps, a.total_calories,
               sp.avg_spo2,
               u.readiness_rating, u.mood, u.energy, u.stress, u.soreness
        FROM sleep s
        LEFT JOIN readiness r ON s.date = r.date
        LEFT JOIN activity a ON s.date = a.date
        LEFT JOIN spo2 sp ON s.date = sp.date
        LEFT JOIN user_metrics u ON s.date = u.date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df = df.sort_values("date")
    df.fillna(method="ffill", inplace=True)
    return df

# --------------------------
# Helper Functions: Mapping Slider Values to Descriptive Text
# --------------------------
def map_readiness(value):
    if value <= 2:
        return "Awful"
    elif value <= 4:
        return "Poor"
    elif value <= 6:
        return "Okay"
    elif value <= 8:
        return "Good"
    else:
        return "Excellent"

def map_mood(value):
    if value <= 2:
        return "Very poor"
    elif value <= 4:
        return "A bit off"
    elif value <= 6:
        return "Okay"
    elif value <= 8:
        return "Good"
    else:
        return "Great"

def map_energy(value):
    if value <= 2:
        return "Exhausted"
    elif value <= 4:
        return "Tired"
    elif value <= 6:
        return "Okay"
    elif value <= 8:
        return "Energized"
    else:
        return "Amped"

def map_stress(value):
    if value <= 2:
        return "Overwhelmed"
    elif value <= 4:
        return "Stressed"
    elif value <= 6:
        return "Neutral"
    elif value <= 8:
        return "Calm"
    else:
        return "Relaxed"

def map_comfort(value):
    if value <= 2:
        return "In pain"
    elif value <= 4:
        return "Sore"
    elif value <= 6:
        return "Moderate"
    elif value <= 8:
        return "Comfortable"
    else:
        return "No discomfort"

def add_descriptions(df):
    df["readiness_feeling_desc"] = df["readiness_rating"].apply(map_readiness)
    df["mood_desc"] = df["mood"].apply(map_mood)
    df["energy_desc"] = df["energy"].apply(map_energy)
    df["stress_desc"] = df["stress"].apply(map_stress)
    df["soreness_desc"] = df["soreness"].apply(map_comfort)
    if "total_sleep_minutes" in df.columns:
        df["total_sleep_hours"] = df["total_sleep_minutes"] / 60.0
    return df

# --------------------------
# Function to Add Combined Scores as Columns
# --------------------------
def add_combined_scores_columns(df):
    combined_list = []
    for index, row in df.iterrows():
        scores = calculate_combined_scores(row)
        combined_list.append(scores)
    combined_df = pd.DataFrame(combined_list)
    df = df.reset_index(drop=True)
    df_combined = pd.concat([df, combined_df], axis=1)
    return df_combined

# --------------------------
# Combined Scores Calculation Function (without default substitution)
# --------------------------
def calculate_combined_scores(record):
    try:
        OS = record.get("sleep_score")
        OR = record.get("readiness_score")
        UR = record.get("readiness_rating")
        UM = record.get("mood")
        UE = record.get("energy")
        US = record.get("stress")
        UC = record.get("soreness")
        if OS is None or OR is None or UR is None or UM is None or UE is None or US is None or UC is None:
            return None
        CSS = (OS + (UR * 10)) / 2
        CMS = (OR + (UM * 10)) / 2
        CES = (OR + (UE * 10)) / 2
        CSTS = (OR + ((11 - US) * 10)) / 2
        CCS = (OS + (UC * 10)) / 2
        O_avg = (OS + OR) / 2
        S_avg = ((UR * 10) + (UM * 10) + (UE * 10) + ((11 - US) * 10) + (UC * 10)) / 5
        Overall = (O_avg + S_avg) / 2
        return {
            "Combined Sleep Score": CSS,
            "Combined Mood Score": CMS,
            "Combined Energy Score": CES,
            "Combined Stress Score": CSTS,
            "Combined Comfort Score": CCS,
            "Overall Combined Well-Being Score": Overall
        }
    except Exception as e:
        return None

# --------------------------
# LLM Report Generation Function
# --------------------------
def generate_llm_report(df, scores):
    prompt = f"""
User Sleep Report and Well-Being Analysis:
- Sleep Quality: {scores['Combined Sleep Score']:.1f}
- Mood Rating: {scores['Combined Mood Score']:.1f}
- Energy Level: {scores['Combined Energy Score']:.1f}
- Stress Level: {scores['Combined Stress Score']:.1f}
- Physical Comfort: {scores['Combined Comfort Score']:.1f}
- Overall Combined Well-Being Score: {scores['Overall Combined Well-Being Score']:.1f}

Based on these scores, please generate a brief, friendly report summarizing the user's sleep and well-being status and offering personalized suggestions. Begin your response with "Hi", and do not use greetings like "hey user". Answer only questions related to these results.
"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach who only answers questions about the user's sleep and well-being data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {e}"

# --------------------------
# Updated Chatbot Functionality (Chat window at bottom, progress spinner, bot renamed)
# --------------------------
def process_chat():
    chat_placeholder = st.empty()
    chat_content = ""
    for speaker, message in st.session_state.chat_history:
        chat_content += f"**{speaker}:** {message}\n\n"
    chat_placeholder.markdown(chat_content)
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Type your message here...")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            with st.spinner("AI Sleep Coach is thinking..."):
                df_all = add_descriptions(load_data())
                response = answer_question(user_query, df_all)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI Sleep Coach", response))
            chat_content = ""
            for speaker, message in st.session_state.chat_history:
                chat_content += f"**{speaker}:** {message}\n\n"
            chat_placeholder.markdown(chat_content)

def answer_question(query, df):
    df_all = add_descriptions(load_data())
    if df_all.empty:
        return "No data available to provide context for your question."
    query_lower = query.lower()
    weekdays = [day.lower() for day in calendar.day_name]
    requested_day = None
    for day in weekdays:
        if day in query_lower:
            requested_day = day
            break
    if requested_day and "today" in query_lower:
        df_all["weekday"] = pd.to_datetime(df_all["date"]).dt.day_name().str.lower()
        req_records = df_all[df_all["weekday"] == requested_day]
        if not req_records.empty:
            req_record = req_records.sort_values("date", ascending=False).iloc[0].to_dict()
        else:
            req_record = None
        latest_record = df_all.sort_values("date", ascending=False).iloc[0].to_dict()
        if req_record is None:
            context_text = f"For today ({pd.to_datetime(latest_record['date']).strftime('%m/%d/%Y')}): Sleep Quality: {calculate_combined_scores(latest_record)['Combined Sleep Score']:.1f}."
        else:
            req_scores = calculate_combined_scores(req_record)
            latest_scores = calculate_combined_scores(latest_record)
            context_text = (
                f"For {pd.to_datetime(req_record['date']).strftime('%A, %m/%d/%Y')} (requested): Sleep Quality: {req_scores['Combined Sleep Score']:.1f}, "
                f"Mood Rating: {req_scores['Combined Mood Score']:.1f}, Energy Level: {req_scores['Combined Energy Score']:.1f}, "
                f"Stress Level: {req_scores['Combined Stress Score']:.1f}, Physical Comfort: {req_scores['Combined Comfort Score']:.1f}, "
                f"Overall: {req_scores['Overall Combined Well-Being Score']:.1f}. "
                f"For today ({pd.to_datetime(latest_record['date']).strftime('%A, %m/%d/%Y')}): Sleep Quality: {latest_scores['Combined Sleep Score']:.1f}, "
                f"Mood Rating: {latest_scores['Combined Mood Score']:.1f}, Energy Level: {latest_scores['Combined Energy Score']:.1f}, "
                f"Stress Level: {latest_scores['Combined Stress Score']:.1f}, Physical Comfort: {latest_scores['Combined Comfort Score']:.1f}, "
                f"Overall: {latest_scores['Overall Combined Well-Being Score']:.1f}."
            )
    else:
        df_sorted = df_all.sort_values("date", ascending=False)
        if len(df_sorted) >= 2:
            latest_record = df_sorted.iloc[0].to_dict()
            previous_record = df_sorted.iloc[1].to_dict()
            latest_scores = calculate_combined_scores(latest_record)
            prev_scores = calculate_combined_scores(previous_record)
            context_text = (
                f"For {pd.to_datetime(previous_record['date']).strftime('%A, %m/%d/%Y')} (previous): Sleep Quality: {prev_scores['Combined Sleep Score']:.1f}, "
                f"Mood Rating: {prev_scores['Combined Mood Score']:.1f}, Energy Level: {prev_scores['Combined Energy Score']:.1f}, "
                f"Stress Level: {prev_scores['Combined Stress Score']:.1f}, Physical Comfort: {prev_scores['Combined Comfort Score']:.1f}, "
                f"Overall: {prev_scores['Overall Combined Well-Being Score']:.1f}. "
                f"For {pd.to_datetime(latest_record['date']).strftime('%A, %m/%d/%Y')} (latest): Sleep Quality: {latest_scores['Combined Sleep Score']:.1f}, "
                f"Mood Rating: {latest_scores['Combined Mood Score']:.1f}, Energy Level: {latest_scores['Combined Energy Score']:.1f}, "
                f"Stress Level: {latest_scores['Combined Stress Score']:.1f}, Physical Comfort: {latest_scores['Combined Comfort Score']:.1f}, "
                f"Overall: {latest_scores['Overall Combined Well-Being Score']:.1f}."
            )
        else:
            record = df_all.iloc[0].to_dict()
            scores = calculate_combined_scores(record)
            context_text = f"For {pd.to_datetime(record['date']).strftime('%A, %m/%d/%Y')}: Sleep Quality: {scores['Combined Sleep Score']:.1f}."
    
    prompt = f"""
You are an expert sleep coach who only answers questions related to the user's sleep and well-being data.
Here is the context based on my combined scores:
{context_text}

Now answer the following question strictly based on these results:
"{query}"
If the question is not related to my sleep and well-being results, reply with "This question is not related to my sleep results."
"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach who answers questions strictly based on the user's sleep and well-being data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating chat response: {e}"

# --------------------------
# STREAMLIT APP INTERFACE (Sidebar Navigation with "Wellness Dashboard")
# --------------------------
st.set_page_config(page_title="Oura Sleep Coach", layout="wide")
st.sidebar.title("Wellness Dashboard")
if "last_updated" in st.session_state:
    st.sidebar.markdown(f"**Last updated:** {st.session_state.last_updated} ✅")
page = st.sidebar.radio("Navigation", 
    ["Your Daily Sleep & Wellness Check-In", 
     "Your AI-Powered Overview & Insights", 
     "Chat with Your AI Sleep Coach", 
     "Data-Driven Insights", 
     "Your Sleep Data Archive"])

if "objective_data_updated" not in st.session_state:
    update_oura_data()
    st.session_state.objective_data_updated = True

init_db()

if page == "Your Daily Sleep & Wellness Check-In":
    col_header, col_date = st.columns([3,1])
    with col_header:
        st.header("Your Daily Sleep & Wellness Check-In")
    with col_date:
        selected_date = st.date_input("Select Date", value=today, min_value=today - timedelta(days=30), max_value=today)
    st.write(f"Selected Date: {selected_date.strftime('%m/%d/%Y')}")
    with st.form("metrics_form"):
        r_input = st.slider("Sleep Readiness (1 = Awful, 10 = Excellent)", min_value=1, max_value=10, value=5, step=1)
        m_input = st.slider("Mood (1 = Very poor, 10 = Great)", min_value=1, max_value=10, value=5, step=1)
        e_input = st.slider("Energy (1 = Exhausted, 10 = Amped)", min_value=1, max_value=10, value=5, step=1)
        s_input = st.slider("Stress (1 = Overwhelmed, 10 = Relaxed)", min_value=1, max_value=10, value=5, step=1)
        so_input = st.slider("Physical Comfort (1 = In pain, 10 = No discomfort)", min_value=1, max_value=10, value=5, step=1)
        submitted = st.form_submit_button("Submit Your Check-In")
    if submitted:
        selected_date_iso = selected_date.strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO user_metrics (date, readiness_rating, mood, energy, stress, soreness)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (selected_date_iso, r_input, m_input, e_input, s_input, so_input))
            conn.commit()
            conn.close()
            st.success("Your daily check-in has been saved!")
            update_oura_data()
        except Exception as e:
            st.error(f"Error saving your check-in: {e}")

elif page == "Your AI-Powered Overview & Insights":
    col_header, col_date = st.columns([3, 1])
    with col_header:
        st.header("Your AI-Powered Overview & Insights")
    with col_date:
        df_all = add_descriptions(load_data())
        available_dates = sorted(df_all["date"].unique())
        if available_dates:
            selected_result_date = st.selectbox("Select Date", available_dates,
                                                index=len(available_dates)-1,
                                                format_func=lambda d: pd.to_datetime(d).strftime("%m/%d/%Y"))
        else:
            selected_result_date = None
    st.write("View your daily combined scores and insights derived from your Oura data and personal check-in. Get a comprehensive overview of your sleep performance and personalized recommendations.")
    if available_dates:
        record = df_all[df_all["date"] == selected_result_date]
        if record.empty:
            st.write(f"No data available for {pd.to_datetime(selected_result_date).strftime('%m/%d/%Y')}.")
        else:
            record = record.iloc[0].to_dict()
            scores = calculate_combined_scores(record)
            if scores is None:
                st.write("Combined scores cannot be computed due to missing data.")
            else:
                overall = scores['Overall Combined Well-Being Score']
                st.markdown(
                    f"<div style='font-size:32px; font-weight:bold; text-align:center; color:#2c3e50;'>Well-Being: {overall:.1f} / 100</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='text-align:center; font-size:16px;'>"
                    f"<strong>Sleep Quality:</strong> {scores['Combined Sleep Score']:.1f} | "
                    f"<strong>Mood Rating:</strong> {scores['Combined Mood Score']:.1f} | "
                    f"<strong>Energy Level:</strong> {scores['Combined Energy Score']:.1f} | "
                    f"<strong>Stress Level:</strong> {scores['Combined Stress Score']:.1f} | "
                    f"<strong>Physical Comfort:</strong> {scores['Combined Comfort Score']:.1f}"
                    f"</div>", unsafe_allow_html=True)
                with st.spinner("Generating your personalized insights..."):
                    llm_report = generate_llm_report(df_all, scores)
                st.subheader("Your Personalized Insights")
                st.write(llm_report)
    else:
        st.write("No data available.")

elif page == "Chat with Your AI Sleep Coach":
    st.header("Chat with Your AI Sleep Coach")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_placeholder = st.empty()
    chat_content = ""
    for speaker, message in st.session_state.chat_history:
        chat_content += f"**{speaker}:** {message}\n\n"
    chat_placeholder.markdown(chat_content)
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Type your message here...")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            with st.spinner("AI Sleep Coach is thinking..."):
                df_all = add_descriptions(load_data())
                response = answer_question(user_query, df_all)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI Sleep Coach", response))
            chat_content = ""
            for speaker, message in st.session_state.chat_history:
                chat_content += f"**{speaker}:** {message}\n\n"
            chat_placeholder.markdown(chat_content)

elif page == "Data-Driven Insights":
    st.header("Data-Driven Insights")
    df_all = add_combined_scores_columns(add_descriptions(load_data()))
    df_all["date"] = pd.to_datetime(df_all["date"])
    fig_line = px.line(df_all, x="date", y=["Combined Sleep Score", "Combined Mood Score", "Combined Energy Score",
                                              "Combined Stress Score", "Combined Comfort Score", "Overall Combined Well-Being Score"],
                       title="Combined Scores Over Time",
                       labels={"value": "Score", "variable": "Combined Score"})
    st.plotly_chart(fig_line, use_container_width=True)
    df_all["weekday"] = df_all["date"].dt.day_name()
    avg_overall = df_all.groupby("weekday")["Overall Combined Well-Being Score"].mean().reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_overall["weekday"] = pd.Categorical(avg_overall["weekday"], categories=order, ordered=True)
    avg_overall = avg_overall.sort_values("weekday")
    fig_bar = px.bar(avg_overall, x="weekday", y="Overall Combined Well-Being Score",
                     title="Average Overall Combined Well-Being Score by Day of Week",
                     labels={"Overall Combined Well-Being Score": "Overall Score", "weekday": "Day of Week"},
                     text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    fig_scatter = px.scatter(df_all, x="Combined Sleep Score", y="Overall Combined Well-Being Score",
                             color="weekday",
                             title="Combined Sleep Score vs Overall Well-Being Score",
                             labels={"Combined Sleep Score": "Combined Sleep Score", 
                                     "Overall Combined Well-Being Score": "Overall Well-Being Score"},
                             trendline=trendline_option,
                             text=df_all["date"].dt.strftime("%m/%d"))
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Your Sleep Data Archive":
    st.header("Your Sleep Data Archive")
    df_all = add_descriptions(load_data())
    df_all["date"] = pd.to_datetime(df_all["date"])
    cutoff_date = pd.to_datetime(today) - timedelta(days=14)
    df_last14 = df_all[df_all["date"] >= cutoff_date]
    if df_last14.empty:
        st.write("No data available for the past 14 days.")
    else:
        st.dataframe(df_last14.sort_values("date").reset_index(drop=True))