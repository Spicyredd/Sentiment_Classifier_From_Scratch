# streamlit_app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image

# Make sure you have these helper files
from CustomModel import TransformerEncoder
from utils import get_text_tensor, get_label

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --- MODEL LOADING ---


@st.cache_resource
def load_model():
    """
    Loads the pre-trained TransformerEncoder model.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    model = TransformerEncoder(
        d_model=768,
        num_heads=4,
        num_layers=2,
        num_classes=1,
        dropout_rate=0.1
    )
    # Load the state dict, ensuring it's mapped to CPU
    model.load_state_dict(torch.load(
        'model_state_dict.pt', map_location='cpu'))
    model.eval()  # Set the model to evaluation mode
    return model


model = load_model()

# --- SIDEBAR ---
st.sidebar.title("About the Model")
st.sidebar.info(
    "This app uses a custom Transformer Encoder model to classify movie review sentiment "
    "as either **Positive** or **Negative**."
)

st.sidebar.header("Model Performance")
try:
    # Display performance metric images
    cm_image = Image.open('visualizations/ConfusionMatrix.png')
    st.sidebar.image(cm_image, caption="Confusion Matrix on Test Data")

    pr_image = Image.open('visualizations/PRCurve.png')
    st.sidebar.image(pr_image, caption="Precision-Recall Curve")

    roc_image = Image.open('visualizations/ROCCurve.png')
    st.sidebar.image(roc_image, caption="ROC Curve")
except FileNotFoundError:
    st.sidebar.error(
        "Performance metric images not found. Make sure 'visualizations/confusion_matrix.png' and 'visualizations/pr_curve.png' exist.")


with st.sidebar.expander("Model Architecture"):
    st.write({
        "Model Type": "Custom Transformer Encoder",
        "d_model": 768,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 1,
        "dropout_rate": 0.1
    })

st.sidebar.markdown("---")
st.sidebar.write("Created by Rishav Beejukchhen")

# --- MAIN PAGE ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown(
    "Enter a movie review below to test the model's sentiment prediction. "
    "You can also select one of the examples."
)

# --- EXAMPLE REVIEWS ---
example_reviews = {
    "Positive Example": "This is a masterpiece of cinema. The acting was superb, the plot was engaging, and the cinematography was breathtaking. I was on the edge of my seat the entire time. A must-see!",
    "Negative Example": "What a disappointment. The plot was predictable and full of holes, the characters were one-dimensional, and the ending felt rushed and unsatisfying. I wouldn't recommend wasting your time on this one.",
    "Neutral/Complex Example": "The film had its moments, with some stunning visual effects, but it was ultimately let down by a weak script. It's a mixed bag that might entertain some viewers but will leave others wanting more."
}

selected_example = st.selectbox(
    "Choose an example to get started:",
    options=list(example_reviews.keys()),
    index=None,  # Makes it blank by default
    placeholder="Select an example review..."
)

with st.form("review_form"):
    initial_text = example_reviews.get(selected_example, "")
    user_input = st.text_area("Enter your movie review here:", value=initial_text,
                              height=150, placeholder="e.g., 'The movie was fantastic!'")

    # Submit button for the form
    submitted = st.form_submit_button("Analyze Sentiment")

# --- PREDICTION LOGIC ---
if submitted and user_input:
    st.markdown("---")
    st.subheader("Analysis Result")
    try:
        # Show a spinner while processing
        with st.spinner("🤖 The model is thinking..."):
            # Convert text to tensor
            tensor = get_text_tensor(user_input)
            # Get model prediction
            raw_output = model(tensor)
            # Process output to get probability and label
            probs, label = get_label(raw_output)

        # Display results with flair
        if label == "pos":
            st.success(f"Sentiment: **Positive**")
        else:
            st.error(f"Sentiment: **Negative**")

        st.write(f"Confidence:")
        st.progress(probs / 100)  # Display probability as a progress bar
        st.write(f"**{probs:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
elif submitted:
    st.warning("Please enter a review before analyzing.")
