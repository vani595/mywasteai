import os
# Hide the annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf

# 1. Choose the right labels file
if os.path.exists("labels.txt"):
    label_path = "labels.txt"
elif os.path.exists("labels"):
    label_path = "labels"
else:
    print("❌ ERROR: Could not find 'labels' or 'labels.txt'!")
    exit()

# 2. Load the Lite Brain
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = open(label_path, "r").readlines()

# 3. Open Camera (Added cv2.CAP_DSHOW to fix that MSMF warning)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f"✅ AI Started! Using {label_path}")
print("Press 'q' on the CAMERA WINDOW to stop.")

while True:
    success, img = cap.read()
    if not success: break

    # Prepare image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
    img_array = (img_array / 127.5) - 1

    # Run AI
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    index = np.argmax(prediction)
    name = class_names[index].strip()

    # Show result
    cv2.putText(img, f"Scanning: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste AI", img)

    # CRITICAL: Click the camera window before pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()