import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Bird Species Classification",
    page_icon="🦜",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

.main{
background:#f4f8fb;
}

.stButton>button{
background:linear-gradient(90deg,#00b09b,#96c93d);
color:white;
border:none;
border-radius:10px;
height:50px;
font-size:18px;
font-weight:bold;
width:100%;
}

.stButton>button:hover{
background:linear-gradient(90deg,#11998e,#38ef7d);
}

.title{
font-size:40px;
font-weight:bold;
text-align:center;
color:#2E8B57;
}

.subtitle{
font-size:20px;
text-align:center;
color:gray;
}

.prediction{
padding:20px;
background:#E8F5E9;
border-radius:15px;
font-size:25px;
font-weight:bold;
color:#2E7D32;
text-align:center;
}

.confidence{
padding:15px;
background:#FFF8E1;
border-radius:15px;
font-size:22px;
text-align:center;
color:#E65100;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bird_species_model.h5")

model = load_model()

# -------------------------------
# Load Class Names
# -------------------------------
with open("class_names.pkl", "rb") as f:
    class_indices = pickle.load(f)

labels = {v: k for k, v in class_indices.items()}

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🦜 Bird Species AI")

st.sidebar.info("""
### Deep Learning Project

**Model**
CNN

**Framework**
TensorFlow / Keras

**Frontend**
Streamlit

**Dataset**
Indian Birds Dataset

Upload a bird image to identify its species.
""")

# -------------------------------
# Header
# -------------------------------
st.markdown("<div class='title'>🦜 Bird Species Classification</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Deep Learning using CNN | TensorFlow | Streamlit</div>", unsafe_allow_html=True)

st.write("")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Bird Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    bird_name = labels[predicted_class]

    with col2:

        st.markdown(
            f"<div class='prediction'>Prediction<br><br>{bird_name}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='confidence'>Confidence : {confidence*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.progress(confidence)

    st.write("")

    st.subheader("Prediction Probabilities")

    df = pd.DataFrame({
        "Bird Species": labels.values(),
        "Probability": prediction[0]
    })

    df = df.sort_values("Probability", ascending=False)

    st.bar_chart(df.set_index("Bird Species"))

# -------------------------------
# Footer
# -------------------------------
st.write("---")

st.markdown(
"""
<center>

Developed using ❤️ with TensorFlow & Streamlit

Bird Species Classification Project

</center>
""",
unsafe_allow_html=True
)