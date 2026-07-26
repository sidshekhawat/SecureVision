"""
dashboard.py

Purpose:
Provides web dashboard for SecureVision.

Features:
- View extracted persons
- View enhanced images
- View alerts
- View watchlist matches

Technology:
Streamlit
"""
import streamlit as st
import sqlite3
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="SecureVision",
    page_icon="🚨",
    layout="wide"
)
st_autorefresh(interval=5000, key="refresh")
st.title("🛡️ SecureVision")
st.caption("AI-Powered ATM Surveillance System")
REPORTS = ROOT / "reports"

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE = REPORTS / "securevision.db"

conn = sqlite3.connect(DATABASE)

try:
    df = pd.read_sql_query(
        "SELECT * FROM logs ORDER BY id DESC",
        conn
    )
except Exception as e:
    st.warning(
        f"Unable to load database: {e}"
    )

    df = pd.DataFrame(
        columns=[
            "id",
            "timestamp",
            "person",
            "status",
            "image_path"
        ]
    )
    
search = st.text_input("🔍 Search Person")

if search:
    df = df[df["person"].str.contains(search, case=False, na=False)]

# Top Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Alerts", len(df))

with col2:
    if len(df) > 0:
        st.metric("Latest Person", df.iloc[0]["person"])
    else:
        st.metric("Latest Person", "-")

with col3:
    if len(df) > 0:
         if df.iloc[0]["status"] == "Watchlist Match":
            st.metric("Latest Status", "🔴 Watchlist Match")
         else:
            st.metric("Latest Status", df.iloc[0]["status"])
    else:
            st.metric("Latest Status", "-")

with col4:
    if len(df) > 0:
        st.metric("Last Detection", df.iloc[0]["timestamp"])
    else:
        st.metric("Last Detection", "--")

st.divider()

# Latest Detection
if len(df) > 0:

    latest = df.iloc[0]

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Latest Detection")
        st.write(f"**Name:** {latest['person']}")
        st.write(f"**Time:** {latest['timestamp']}")
        st.write(f"**Status:** {latest['status']}")

    with right:
        if latest["image_path"] and latest["image_path"] != "None":
            try:
                image_path = Path(latest["image_path"])

                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption="Captured Evidence"
                    )
            except Exception as e:
                st.warning(
                    f"Unable to display image: {e}"
                )

st.divider()

st.subheader("Detection History")
st.dataframe(df, use_container_width=True)

conn.close()