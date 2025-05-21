from fastapi import FastAPI, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import json

from .auth import get_api_key
from .api_utils import image_from_upload_file
from .ai.ocr import GeminiOCR
from .ai.detector import GeminiDetector


app = FastAPI(title="Zawsze lubiłem dżem", dependencies=[Depends(get_api_key)])


gemini_ocr = GeminiOCR()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    image = await image_from_upload_file(file)
    text = gemini_ocr.ocr(image)
    return JSONResponse(content={
        "text": text
    })


gemini_detector = GeminiDetector(model_name="gemini-2.5-flash-preview-04-17")

class DetectRequest(BaseModel):
    labels: list[str]
    descriptions: list[str]

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    labels: str = Form(...),
    descriptions: str = Form(...)
):
    image = await image_from_upload_file(file)
    labels_list = json.loads(labels)
    descriptions_list = json.loads(descriptions)
    detections = gemini_detector.detect(image, labels_list, descriptions_list)
    return JSONResponse(content=detections)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
