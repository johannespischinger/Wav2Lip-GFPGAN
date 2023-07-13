import json
import os
import requests
import boto3


def get_generate_file(audio_file: str, video_file: str, video_name: str) -> None:
    """Triggers the generation of the lipsynced video for a given audio and video file.
    The created lypsinc video is saved then in the s3 bucket "dev.susio.videocreation", folder /cache/video_name".

    Args:
        audio_file (str): Path to the audio file.
        video_file (str): Path to the video file.
        video_name (str): Name of the video file.

    Raises:
        Exception: In case if the API call failed

    Returns:
        None: None
    """
    try:
        print(f"\nprocessing '{audio_file.split('/')[-1]}'... ")
        url = "http://127.0.0.1:8000/inference" #Adjust IP address here

        files = [
            ("audio", (audio_file, open(audio_file, "rb"), "audio/wav")),
            ("video", (video_file, open(video_file, "rb"), "video/mp4")),
        ]
        headers = {
            os.environ['API_KEY_NAME']: os.environ['INFERENCE_API_KEY'],

        }
        data = {
            "video_name": video_name,
        }

        # note: for multipart form-data, no header is needed
        response = requests.post(url, files=files, data=data, headers=headers)

        if response.status_code != 200:
            print(response.status_code, response.reason, response.text)
            raise Exception(f"Could not connect to API for {audio_file} - {video_file}")

    except Exception as e:
        print(f"error: {e}")
        raise Exception(f"could not connect to API for {audio_file} - {video_file}")


def handler(event, context):
    """Lambda handler for batch lip syncing.
    This function is triggered when files are inserted into an S3 bucket and
    sends the respective files to the lip-syncing API

    Args:
        event (_type_): _description_
        context (_type_): _description_

    Returns:
        _type_: _description_
    """
    os.chdir("/tmp/")

    s3_client = boto3.client('s3')

    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        key_list = key.split("/")
        s3_client.download_file(bucket, key, key_list[4])

    # key_list:
    # [0] - userID
    # [1] - speakerID
    # [2] - campaignID
    # [3] - raw-folder
    # [4] - filename

    video = 'input_video.mp4'
    s3_client.download_file(bucket, f'{key_list[0]}/{key_list[1]}/{key_list[2]}/raw/input_video.mp4', video)

    name = key_list[4].replace(".wav", ".mp4")
    out_name = f"{key_list[0]}-{key_list[1]}-{key_list[2]}-{name}"

    # start lip syncing process
    get_generate_file(audio_file=key_list[4], video_file=video, video_name=out_name)

    body = {
        "message": f"Video {out_name} created successfully",
        "input": event,
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response


# event = {'Records': [{'eventVersion': '2.1', 'eventSource': 'aws:s3', 'awsRegion': 'eu-central-1', 'eventTime': '2023-06-28T07:54:34.149Z', 'eventName': 'ObjectCreated:Put', 'userIdentity': {'principalId': 'AWS:AIDA2XEJBWKGRJRQRBWQS'}, 'requestParameters': {'sourceIPAddress': '24.134.130.189'}, 'responseElements': {'x-amz-request-id': '6RY4Z0X9FFVVS9YF', 'x-amz-id-2': '4CHasCb6sl9TtNrns22Bierj/hUUfmGuI1uUyONqR372BEghfxNiNuoUW+eQTuF75NPnXdTkcuzrGbLE5JjYm+bNosEkK39W'}, 's3': {'s3SchemaVersion': '1.0', 'configurationId': 'susio-dev-audiotransfer-3fa24e0836e79f7bd46874ffab17b742', 'bucket': {'name': 'dev.susio.videogeneration', 'ownerIdentity': {'principalId': 'A2BPFWF8CFD43W'}, 'arn': 'arn:aws:s3:::dev.susio.videogeneration'}, 'object': {'key': 'UID001/SID006/CID017/audio/CID017_Mohammed.wav', 'size': 441102, 'eTag': 'f429108f646378075138b529f9426c1b', 'sequencer': '00649BE739DBC947A8'}}}]}
#
#
# if __name__ == "__main__":
#     handler(event, None)

