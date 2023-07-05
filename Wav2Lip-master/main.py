import os
import shutil
from http.client import HTTPException

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.security import APIKeyHeader

from pydantic import BaseModel
from starlette.status import HTTP_403_FORBIDDEN

from inference import Inference
import boto3

app = FastAPI()

# Define your API keys
load_dotenv()
API_KEYS = [os.getenv('INFERENCE_API_KEY')]

# Define the API key name and location (e.g., header, query parameter, cookie)
API_KEY_NAME = os.getenv('API_KEY_NAME')
API_KEY_LOCATION = APIKeyHeader(name=API_KEY_NAME)


# API key dependency
async def verify_api_key(api_key: str = Depends(API_KEY_LOCATION)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")
    return api_key


class LipSync(BaseModel):
    videoURL: str


@app.get("/")
async def root():
    return {"message": "Hello sweetheart, lets upload some files to the API"}


@app.post("/inference", dependencies=[Depends(verify_api_key)])
async def inference(audio: UploadFile = File(...), video: UploadFile = File(...), video_name: str = Form(...)):
    # Save the video file
    video_filename = f"inputs/{video.filename}"
    with open(video_filename, "wb") as f:
        f.write(await video.read())

    # Save the audio file
    audio_filename = f"inputs/{audio.filename}"
    with open(audio_filename, "wb") as f:
        f.write(await audio.read())

    # Add your processing code here

    # Set the default output file name if video_name is not provided
    if video_name:
        output_file_name = video_name
    else:
        output_file_name = "output"

    # Set the output file path
    output_file_path = f"results/{output_file_name}"
    inference = Inference(video=video_filename, audio=audio_filename, outputFile=output_file_path)
    output_file = inference.run()

    # Rename the file if a file name is provided
    # if video_name:
    #     new_file_path = f"uploads/{video_name}"
    #     shutil.move(output_file, new_file_path)
    #     output_file = new_file_path

    s3_bucket_name = "dev.susio.videogeneration"
    s3_key = f"cache/{video_name}"
    s3_client = boto3.client("s3")
    s3_client.upload_file(output_file, s3_bucket_name, s3_key)

    # Generate the S3 URL for the uploaded video
    video_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"

    # Delete the local files
    os.remove(video_filename)
    os.remove(audio_filename)

    return {
        "message": "File uploaded successfully",
        "video_url": video_url
    }
    # LipSync(video_url=video_url)

    # with open(fileLocation, "wb+") as file_object:
    #     file_object.write(fileLocation)
    # return {'message': FileResponse(fileLocation, media_type="video/mp4")}
    # return {"info": f"file '{outputFileName}' saved at '{fileLocation}'"}

    # # Return the processed filenames
    # filename = os.path.basename(urlOutputFile)
    # headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
    #
    # return {'message': FileResponse(urlOutputFile, headers=headers, media_type="video/mp4")}
