from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from .auth import get_api_key
from .api_utils import image_from_upload_file
from .ai.ocr import GeminiOCR


app = FastAPI(title="Zawsze lubiłem dżem", dependencies=[Depends(get_api_key)])


gemini_ocr = GeminiOCR()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    image = await image_from_upload_file(file)
    text = gemini_ocr.ocr(image)
    return JSONResponse(content={
        "text": text
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
