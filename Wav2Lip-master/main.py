from pydantic import BaseModel
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


class VideoInference(BaseModel):
    awsURL: str
    audioID: str


@app.get("/")
async def root():
    return {"message": "Hello sweetheart, lets upload some files to the API"}


@app.post("/inference", dependencies=[Depends(verify_api_key)])
def inference(event: dict):

    run_inference(event)

    return {
        "message": "Inference triggered successfully",
    }


def run_inference(event):

    # Init the S3 client
    s3_client = boto3.client('s3')

    # Get the bucket name and other file information
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        key_list = key.split("/")

    # key_list:
    # [0] - userID
    # [1] - speakerID
    # [2] - campaignID
    # [3] - raw-folder
    # [4] - filename

    videoFile = f'inputs/input_video.mp4'
    audioFile = f'inputs/{key_list[4]}'

    # Download the audio file
    s3_client.download_file(bucket, key, f'inputs/{key_list[4]}')

    s3_client.download_file(bucket, f'{key_list[0]}/{key_list[1]}/{key_list[2]}/raw/input_video.mp4',videoFile)

    # Define the output file name
    name = key_list[4].replace(".wav", ".mp4")
    video_name = f"{key_list[0]}-{key_list[1]}-{key_list[2]}-{name}"

    # Set the output file path
    output_file_path = f"results/{video_name}"

    # Run inference and GAN
    inference = Inference(video=videoFile, audio=audioFile, outputFile=output_file_path)
    output_file_path = inference.run()
    gan = GanInference(videoPath=output_file_path, audioPath=audioFile)
    gan.run()

    # Defining the s3 upload with bucket name and file name
    s3_bucket_name = "dev.susio.videogeneration"
    s3_key = f"cache/{video_name}"
    s3_client = boto3.client("s3")
    s3_client.upload_file(output_file_path, s3_bucket_name, s3_key)

    model = VideoInference(
        awsURL=f"https://{s3_bucket_name}/{s3_key}",
        audioID=f"{key_list[4]}"
    )

    # Generate the S3 URL for the uploaded video

    # Delete the local files
    os.remove(videoFile)
    os.remove(audioFile)
    os.remove(output_file_path)

    return model
