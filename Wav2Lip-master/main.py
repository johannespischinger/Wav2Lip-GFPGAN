import os
from http.client import HTTPException

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.security import APIKeyHeader

from starlette.status import HTTP_403_FORBIDDEN

from inference import Inference
from ganInference import GanInference
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

    # Set the default output file name if video_name is not provided
    if video_name:
        output_file_name = video_name
    else:
        output_file_name = "output"

    # Set the output file path
    output_file_path = f"results/{output_file_name}"

    # Run inference and GAN
    inference = Inference(video=video_filename, audio=audio_filename, outputFile=output_file_path)
    output_file_path = inference.run()
    gan = GanInference(videoPath=output_file_path, audioPath=audio_filename)
    gan.run()

    # Defining the s3 upload with bucket name and file name
    s3_bucket_name = "dev.susio.videogeneration"
    s3_key = f"cache/{video_name}"
    s3_client = boto3.client("s3")
    s3_client.upload_file(output_file_path, s3_bucket_name, s3_key)

    # Generate the S3 URL for the uploaded video
    video_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"

    # Delete the local files
    os.remove(video_filename)
    os.remove(audio_filename)
    os.remove(output_file_path)

    return {
        "message": "File uploaded successfully",
        "video_url": video_url
    }