from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
from fastapi.responses import StreamingResponse

app = FastAPI()

model = YOLO('best.pt')  # Confirm correct model path

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Ensure input size matches model training
    image_np = np.array(image)  # Convert to NumPy array

    # Perform object detection
    results = model(image_np)

    # Draw bounding boxes
    result_image = results[0].plot()

    # Convert result image to PIL for response
    result_image_pil = Image.fromarray(result_image)
    buffer = BytesIO()
    result_image_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
