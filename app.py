import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import re
from profanity import english_profanity_checker, process_file
from privacy import PrivacyComplianceDetector
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Call Center Analytics Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1.0rem;
        color: #424242;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .warning {
        color: #ff4444;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.markdown(
    "<h1 class='main-header'>Call Center Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='info-text'>Upload call transcript files for analysis of profanity, compliance violations.</p>",
    unsafe_allow_html=True,
)

# Sidebar for navigation and options
st.sidebar.markdown("## Analysis Options")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload JSON Call Transcript", type=["json"])
if uploaded_file is not None:
    # Read the file content
    file_content = uploaded_file.read()

    try:
        # Parse the JSON string to a Python dictionary
        data_dict = json.loads(file_content)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")

# Analysis type selector
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Profanity Detection", "Compliance Check", "Speech Overlap Analysis"],
)


# Function to detect overlaps in a call transcript
def detect_overlaps(call_data):
    """Detect overlapping speech segments between Agent and Customer."""
    agent_segments = [
        entry for entry in call_data if entry["speaker"].lower() == "agent"
    ]
    customer_segments = [
        entry for entry in call_data if entry["speaker"].lower() == "customer"
    ]

    overlaps = []
    for a in agent_segments:
        for c in customer_segments:
            if (a["stime"] < c["etime"]) and (a["etime"] > c["stime"]):
                overlap_start = max(a["stime"], c["stime"])
                overlap_end = min(a["etime"], c["etime"])
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    overlaps.append(
                        {
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_duration": overlap_duration,
                            "agent_stime": a["stime"],
                            "agent_etime": a["etime"],
                            "customer_stime": c["stime"],
                            "customer_etime": c["etime"],
                            "initiator": "Customer"
                            if c["stime"] > a["stime"]
                            else "Agent",
                        }
                    )
    return overlaps


# Profanity analysis visualization function
def display_profanity_results(profanity_data):
    st.markdown(
        "<h2 class='sub-header'>Profanity Detection Results</h2>",
        unsafe_allow_html=True,
    )

    if len(profanity_data) == 0:
        st.info("No profanity detected in the transcript.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(profanity_data)

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Profanity Instances", len(df))
    with col2:
        st.metric("Unique Profane Terms", df["profane_term"].nunique())
    with col3:
        st.metric("Speakers Using Profanity", df["speaker"].nunique())

    # Display detailed table
    st.markdown("### Detected Profanity")
    st.dataframe(df)

    # Display charts
    col1, col2 = st.columns(2)

    with col1:
        # Profanity by speaker
        speaker_counts = df["speaker"].value_counts().reset_index()
        speaker_counts.columns = ["Speaker", "Count"]
        fig = px.pie(
            speaker_counts,
            values="Count",
            names="Speaker",
            title="Profanity Distribution by Speaker",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top profane terms
        term_counts = df["profane_term"].value_counts().reset_index()
        term_counts.columns = ["Term", "Count"]
        term_counts = term_counts.head(10)  # Top 10
        fig = px.bar(
            term_counts,
            x="Term",
            y="Count",
            title="Most Frequent Profane Terms",
            color="Count",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline of profanity
    fig = px.scatter(
        df,
        x="timestamp_start",
        y="speaker",
        color="profane_term",
        size=[10] * len(df),
        title="Timeline of Profanity in Call",
        labels={"timestamp_start": "Time (seconds)", "speaker": "Speaker"},
    )
    st.plotly_chart(fig, use_container_width=True)


# Compliance check visualization function
def display_compliance_results(compliance_data):
    st.markdown(
        "<h2 class='sub-header'>Compliance Check Results</h2>", unsafe_allow_html=True
    )

    # Create columns for key metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Verification")
        verified = compliance_data.get("verification_performed", False)
        verification_method = compliance_data.get("verification_method", "None")

        if verified:
            st.markdown("✅ **Verification Performed**")
            st.markdown(f"Method: **{verification_method}**")
        else:
            st.markdown("❌ **No Verification Performed**")

    with col2:
        st.markdown("### Sensitive Information")
        sensitive_shared = compliance_data.get("sensitive_info_shared", False)
        sensitive_type = compliance_data.get("sensitive_info_type", "None")

        if sensitive_shared:
            st.markdown("⚠️ **Sensitive Information Shared**")
            st.markdown(f"Type: **{sensitive_type}**")
        else:
            st.markdown("✅ **No Sensitive Information Shared**")

    # Violation status with appropriate styling
    is_violation = compliance_data.get("is_violation", False)
    if is_violation:
        st.markdown(
            "<div class='highlight warning'><h3>⚠️ COMPLIANCE VIOLATION DETECTED</h3></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='highlight'><h3>✅ No Compliance Violations</h3></div>",
            unsafe_allow_html=True,
        )

    # Explanation
    explanation = compliance_data.get("explanation", "No explanation provided")
    st.markdown("### Analysis Explanation")
    st.markdown(f"{explanation}")

    # Call ID
    call_id = compliance_data.get("call_id", "Unknown")
    st.markdown(f"**Call ID:** {call_id}")


# Overlap analysis visualization function
def display_overlap_results(call_data):
    st.markdown(
        "<h2 class='sub-header'>Speech Overlap Analysis</h2>", unsafe_allow_html=True
    )

    # Detect overlaps
    overlaps = detect_overlaps(call_data)

    if not overlaps:
        st.info("No speech overlaps detected in this call.")
        return

    # Convert to DataFrame for easier manipulation
    overlap_df = pd.DataFrame(overlaps)

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Overlaps", len(overlaps))
    with col2:
        st.metric(
            "Total Overlap Duration", f"{overlap_df['overlap_duration'].sum():.2f}s"
        )
    with col3:
        st.metric(
            "Avg Overlap Duration", f"{overlap_df['overlap_duration'].mean():.2f}s"
        )

    # Determine call end time
    call_end_time = max([entry["etime"] for entry in call_data])

    # 1. Overlap Duration Over Time (similar to third image)
    st.markdown("### Overlapping Speech Duration Over Time")

    fig = px.bar(
        overlap_df,
        x="overlap_start",
        y="overlap_duration",
        title="Overlapping Speech Duration Between Agent and Customer Over Time",
        labels={
            "overlap_start": "Time (seconds)",
            "overlap_duration": "Overlap Duration (seconds)",
        },
        color_discrete_sequence=["purple"],
    )

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Overlap Duration (seconds)",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray"),
        yaxis=dict(
            gridcolor="lightgray", range=[0, max(overlap_df["overlap_duration"]) * 1.1]
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2. Overlap Frequency by Time Interval (similar to first and second images)
    st.markdown("### Count of Overlapping Speech Occurrences per 5-Second Interval")

    # Create time bins (5-second intervals)
    interval = 5
    bins = list(range(0, int(call_end_time) + interval, interval))
    overlap_df["time_bin"] = pd.cut(overlap_df["overlap_start"], bins=bins, right=False)

    # Count overlaps per bin
    overlap_counts = (
        overlap_df.groupby("time_bin").size().reset_index(name="overlap_count")
    )
    overlap_counts["time_bin_str"] = overlap_counts["time_bin"].astype(str)

    fig = px.bar(
        overlap_counts,
        x="time_bin_str",
        y="overlap_count",
        title=f"Count of Overlapping Speech Occurrences per {interval}-Second Interval",
        labels={
            "time_bin_str": "Time Interval (seconds)",
            "overlap_count": "Number of Overlapping Speech Occurrences",
        },
        color_discrete_sequence=["teal"],
    )

    fig.update_layout(
        xaxis_title="Time Interval (seconds)",
        yaxis_title="Number of Overlapping Speech Occurrences",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", tickangle=45),
        yaxis=dict(gridcolor="lightgray"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Initiator distribution (pie chart)
    st.markdown("### Distribution of Overlap Initiators")

    initiator_counts = overlap_df["initiator"].value_counts().reset_index()
    initiator_counts.columns = ["initiator", "count"]

    fig = px.pie(
        initiator_counts,
        values="count",
        names="initiator",
        title="Distribution of Overlap Initiators",
        color_discrete_sequence=["lightblue", "lightcoral"],
    )

    fig.update_layout(showlegend=True, legend_title="Initiator")

    st.plotly_chart(fig, use_container_width=True)

    # 4. Overlap data table
    with st.expander("View Raw Overlap Data"):
        st.dataframe(overlap_df)


# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Display relevant analysis based on selection
        if analysis_type == "Profanity Detection":
            st.markdown(
                "<h2 class='sub-header'>Profanity Detection</h2>",
                unsafe_allow_html=True,
            )

            # Option to use LLM
            use_llm = st.sidebar.checkbox("Use LLM for Enhanced Detection", value=False)

            # Simulate profanity results with sample data
            if st.button("Run Profanity Analysis"):
                with st.spinner("Analyzing profanity in transcript..."):
                    # Display sample profanity results
                    prof_res = process_file(
                        filepath="", file_upload=data_dict, upload=True
                    )
                    display_profanity_results(prof_res)

        elif analysis_type == "Compliance Check":
            st.markdown(
                "<h2 class='sub-header'>Compliance Check</h2>", unsafe_allow_html=True
            )

            # Simulate compliance results with sample data
            if st.button("Run Compliance Check"):
                with st.spinner("Analyzing transcript for compliance violations..."):
                    # Display sample compliance results
                    # sample_compliance_data = {
                    #     "verification_performed": False,
                    #     "verification_method": "None",
                    #     "sensitive_info_shared": False,
                    #     "sensitive_info_type": "None",
                    #     "is_violation": False,
                    #     "explanation": "The agent did not share any sensitive information, and there was no verification of the customer's identity performed before any such sharing could occur. The conversation was terminated after the customer denied having an account with Definite Bank, and no sensitive information was disclosed.",
                    #     "call_id": "00be25b0-458f-4cbf-ae86-ae2ec1f7fba4.json",
                    # }
                    compliance_checker = PrivacyComplianceDetector(
                        api_key=os.environ.get("GROQ_API_KEY"),
                    )

                    comp_res = compliance_checker.process_single_file(
                        file_path=data_dict, upload=True
                    )

                    print(comp_res)

                    display_compliance_results(comp_res)

        elif analysis_type == "Speech Overlap Analysis":
            # For overlap analysis, use the actual implementation
            if st.button("Run Overlap Analysis"):
                with st.spinner("Analyzing speech overlaps in transcript..."):
                    display_overlap_results(data_dict)

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a JSON file to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    "<p class='info-text'>Call Center Analytics Dashboard | &copy; 2025</p>",
    unsafe_allow_html=True,
)
