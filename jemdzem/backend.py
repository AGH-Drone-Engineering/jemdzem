from fastapi import FastAPI, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import json

from .auth import get_api_key
from .api_utils import image_from_upload_file
from .ai.ocr import GeminiOCR
from .ai.multi_detector import GeminiMultiDetector
from .ai.single_detector import GeminiSingleDetector


app = FastAPI(title="Zawsze lubiłem dżem", dependencies=[Depends(get_api_key)])


gemini_ocr = GeminiOCR()

@app.post("/ocr")
async def api_ocr(file: UploadFile = File(...)):
    image = await image_from_upload_file(file)
    text = gemini_ocr.ocr(image)
    return JSONResponse(content={
        "text": text
    })


gemini_multi_detector = GeminiMultiDetector()

class MultiDetectRequest(BaseModel):
    labels: list[str]
    descriptions: list[str]

@app.post("/multi-detect")
async def api_multi_detect(
    file: UploadFile = File(...),
    labels: str = Form(...),
    descriptions: str = Form(...),
    model_name: str = "gemini-2.0-flash",
):
    image = await image_from_upload_file(file)
    labels_list = json.loads(labels)
    descriptions_list = json.loads(descriptions)
    detections = gemini_multi_detector.detect(image, labels_list, descriptions_list, model_name)
    return JSONResponse(content=detections)


gemini_single_detector = GeminiSingleDetector()

class SingleDetectRequest(BaseModel):
    label: str
    description: str

@app.post("/single-detect")
async def api_single_detect(
    file: UploadFile = File(...),
    ref_file: UploadFile | None = File(None),
    label: str = Form(...),
    description: str = Form(...),
    model_name: str = "gemini-2.0-flash",
):
    image = await image_from_upload_file(file)
    if ref_file is not None:
        ref_image = await image_from_upload_file(ref_file)
    else:
        ref_image = None
    detections = gemini_single_detector.detect(image, label, description, model_name, ref_image)
    return JSONResponse(content=detections)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
