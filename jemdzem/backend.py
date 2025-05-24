from fastapi import FastAPI, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import json
import os
import numpy as np

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
    ref_files: list[UploadFile] | None = File(None),
    labels: str = Form(...),
    descriptions: str = Form(...),
    model_name: str = "gemini-2.0-flash",
):
    """Detect multiple classes using the single detector internally.

    The API mirrors ``/multi-detect`` but executes a separate single detector
    call for every provided label/description pair. Optional reference images can
    be supplied for individual classes by sending multiple ``ref_file`` form
    fields. The reference image is matched to the label by comparing the file
    name (without extension) with the class label.
    """

    image = await image_from_upload_file(file)
    labels_list = json.loads(labels)
    descriptions_list = json.loads(descriptions)

    # Load reference images into a mapping {label: image}
    ref_map: dict[str, np.ndarray] = {}
    if ref_files:
        for rfile in ref_files:
            label_name, _ = os.path.splitext(rfile.filename)
            ref_map[label_name] = await image_from_upload_file(rfile)

    results = []
    for label, description in zip(labels_list, descriptions_list):
        ref_image = ref_map.get(label)
        detections = gemini_single_detector.detect(
            image, label, description, model_name, ref_image
        )
        for det in detections:
            det_with_label = det.copy()
            det_with_label["label"] = label
            results.append(det_with_label)

    return JSONResponse(content=results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
