import streamlit as st
import pandas as pd
import joblib
import re
import json

model = joblib.load("models/smd_logistic_regression_model.pkl")

SENSATIONAL_WORDS = [
    "breaking", "shocking", "miracle", "secret",
    "urgent", "must share", "viral", "hidden truth"
]

def detect_evidence(text):
    evidence = []
    text_lower = text.lower()

    if any(word in text_lower for word in SENSATIONAL_WORDS):
        evidence.append({
            "signal": "sensational_language",
            "description": "The post contains emotional, urgent, or viral-style words."
        })

    if text.count("!") >= 3:
        evidence.append({
            "signal": "excessive_punctuation",
            "description": "The post contains excessive exclamation marks."
        })

    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if len(caps_words) > 0:
        evidence.append({
            "signal": "all_caps_language",
            "description": "The post contains excessive uppercase emphasis."
        })

    return evidence

def predict_post(post_text):
    input_df = pd.DataFrame([{"text": post_text}])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_names = model.classes_

    class_probabilities = {
        class_names[i]: round(float(probabilities[i]), 3)
        for i in range(len(class_names))
    }

    credibility_score = round(float(max(probabilities)), 3)
    evidence = detect_evidence(post_text)
    classification = prediction

    if credibility_score < 0.55 and len(evidence) > 0:
        classification = "Suspicious"

    return {
        "classification": classification,
        "credibility_score": credibility_score,
        "confidence": f"{round(credibility_score * 100, 2)}%",
        "class_probabilities": class_probabilities,
        "evidence": evidence
    }

st.set_page_config(
    page_title="Social Media Misinformation Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Social Media Misinformation Detector")

st.markdown("""
Enter any social media post and the model will analyze:
- Reliability
- Credibility Score
- Misinformation Signals
- Evidence Report
""")

with st.expander("📝 Example Posts"):
    st.code("""
Breaking News!!! Doctors found a MIRACLE cure. Share urgently!!!

Scientists confirm water boils at 100°C under standard conditions.

Secret government plan exposed! MUST SHARE NOW!!!
""")

post_text = st.text_area(
    "Enter Social Media Post",
    height=200,
    placeholder="Type or paste a social media post here..."
)

if st.button("Analyze Post"):
    if post_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_post(post_text)

        if result["classification"] == "Reliable":
            st.success("✅ Content appears Reliable")
        elif result["classification"] == "Suspicious":
            st.warning("⚠️ Content may contain misinformation signals")
        else:
            st.error("🚨 High risk of misinformation detected")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Classification", result["classification"])

        with col2:
            st.metric("Credibility Score", result["credibility_score"])

        with col3:
            st.metric("Confidence", result["confidence"])

        st.subheader("📊 Class Probabilities")

        prob_df = pd.DataFrame({
            "Class": list(result["class_probabilities"].keys()),
            "Probability": list(result["class_probabilities"].values())
        })

        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("🧾 Evidence Report")
        st.json(result["evidence"])

        report_json = json.dumps(result, indent=4)

        st.download_button(
            label="📥 Download Report",
            data=report_json,
            file_name="misinformation_report.json",
            mime="application/json"
        )

st.markdown("---")
st.caption("Built using Python, Scikit-Learn and Streamlit | SMD Project")