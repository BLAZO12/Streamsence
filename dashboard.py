import pandas as pd
import streamlit as st

st.title("📊 Classroom Attention Dashboard")

df = pd.read_csv("logs/attention_log.csv")

# Overall attentiveness
attentive_rate = (df["Status"] == "Attentive").mean() * 100
st.metric("Overall Attentiveness", f"{attentive_rate:.2f}%")

# Per student stats
student_stats = df.groupby(["Student_ID", "Status"]).size().unstack(fill_value=0)
student_stats["Attention %"] = student_stats["Attentive"] / (student_stats.sum(axis=1)) * 100
st.dataframe(student_stats)

# Timeline chart
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
timeline = df.groupby("Timestamp")["Status"].apply(lambda x: (x=="Attentive").mean()*100)
st.line_chart(timeline)
