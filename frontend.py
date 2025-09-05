import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"  # FastAPI backend

st.set_page_config(page_title="Meeting Intelligence System", layout="wide")

st.title("🤝 Meeting Intelligence Brief Generator")

# --- Input Form ---
st.sidebar.header("🔍 Prospect Information")
attendee_name = st.sidebar.text_input("Attendee Name", "Vinay Krishna Gupta")
title = st.sidebar.text_input("Title", "CEO")
organization = st.sidebar.text_input("Organization", "Antino Labs Private Limited")
meeting_date = st.sidebar.date_input("Meeting Date")
our_company = st.sidebar.text_input("Our Company", "Flipkart")
our_solutions = st.sidebar.text_area(
    "Our Solutions (comma-separated)", "AI Solutions, Digital Transformation"
)

if st.sidebar.button("Generate Brief"):
    with st.spinner("Generating meeting brief... ⏳"):
        payload = {
            "attendee_name": attendee_name,
            "title": title,
            "organization": organization,
            "meeting_date": str(meeting_date),
            "our_company": our_company,
            "our_solutions": [s.strip() for s in our_solutions.split(",")]
        }

        try:
            res = requests.post(f"{API_BASE}/api/generate-brief", json=payload)
            if res.status_code == 200:
                brief = res.json()

                # --- Display Brief ---
                st.success("✅ Brief generated successfully!")
                st.subheader(f"Prospect: {brief['prospect_info']['name']} ({brief['prospect_info']['title']})")
                st.caption(f"Organization: {brief['prospect_info']['organization']}, Meeting Date: {brief['prospect_info']['meeting_date']}")
                st.metric("Confidence Score", f"{brief['confidence_score']*100:.1f}%")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎯 Key Pitch Points")
                    for p in brief["key_pitch_points"]:
                        st.write(f"- {p}")

                    st.markdown("### 📘 Background & Education")
                    for b in brief["background_education"]:
                        st.write(f"- {b}")

                    st.markdown("### 🔗 Connection Opportunities")
                    for c in brief["connection_opportunities"]:
                        st.write(f"- {c}")

                with col2:
                    st.markdown("### 📰 Recent Highlights")
                    for h in brief["recent_highlights"]:
                        st.write(f"- {h}")

                    st.markdown("### 🏢 Portfolio & Departments")
                    for p in brief["portfolio_departments"]:
                        st.write(f"- {p}")

                    st.markdown("### 🚀 Major Initiatives")
                    for i in brief["major_initiatives"]:
                        st.write(f"- {i}")

            else:
                st.error(f"❌ Error: {res.status_code} - {res.text}")

        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")

# --- Sidebar Extras ---
st.sidebar.markdown("---")
if st.sidebar.button("View Recent Briefs"):
    res = requests.get(f"{API_BASE}/api/recent-briefs")
    if res.status_code == 200:
        briefs = res.json()
        st.sidebar.success(f"Found {len(briefs)} briefs")
        for b in briefs:
            st.sidebar.write(f"- {b['person_name']} ({b['organization']}) - {b['created_at']}")
    else:
        st.sidebar.error("Failed to load briefs.")
