import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("♻️ Smart Waste Classifier (TFLite)")

# 1. TFLite Model Load Karne Ka Function
@st.cache_resource
def load_tflite_model():
    # YAHAN DEKHO: Naam wahi likhna jo tumhare folder mein hai
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    
    # Labels load karo
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return interpreter, class_names

try:
    interpreter, class_names = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    file = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # 2. Preprocessing (TFLite input size ke liye)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        normalized_image_array = (image_array / 127.5) - 1
        input_data = np.expand_dims(normalized_image_array, axis=0)

        # 3. Prediction (TFLite Inference)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(output_data[0])
        class_name = class_names[index]
        confidence_score = output_data[0][index]

        st.divider()
        
        # Result clean up
        result_text = class_name.split(" ", 1)[-1] 
        
        if "Non" in result_text or "Plastic" in result_text:
            st.error(f"### Result: {result_text}")
            st.warning("💡 **Tip:** Dispose of this in the **Blue Bin** (Recyclable).")
        else:
            st.success(f"### Result: {result_text}")
            st.info("💡 **Tip:** This is organic waste. Use the **Green Bin** (Compostable).")
            
        # Confidence Level
        st.write(f"**Confidence Level:** {round(confidence_score * 100)}%")
        st.progress(int(confidence_score * 100))

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check karo ki 'model_unquant.tflite' aur 'labels.txt' sahi folder mein hain.")