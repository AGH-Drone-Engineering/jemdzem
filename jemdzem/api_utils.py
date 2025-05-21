from fastapi import UploadFile
import numpy as np
import cv2


async def image_from_upload_file(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
