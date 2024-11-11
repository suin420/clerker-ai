import os
import json
import boto3
from STT.ClovaText import make_stt_txt
from Chunking.EmbeddingChunking import make_chunk

ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
bucket_name = "clerkerai"
region_name = "eu-north-1"

s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
    )

def download_from_s3(s3_path, local_path):
    s3.download_file(bucket_name, s3_path, local_path)

def upload_to_s3(local_path, s3_path):
    s3.upload_file(local_path, bucket_name, s3_path)

def download_folder_from_s3(s3_folder, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue

                relative_path = key[len(s3_folder):].lstrip('/')
                local_file_path = os.path.join(local_dir, relative_path)

                local_file_dir = os.path.dirname(local_file_path)
                os.makedirs(local_file_dir, exist_ok=True)

                s3.download_file(bucket_name, key, local_file_path)


def lambda_handler(event, context):
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

    os.makedirs('/tmp/STT', exist_ok=True)
    os.makedirs('/tmp/STT/stt_audio', exist_ok=True)
    os.makedirs('/tmp/STT/stt_text/KeywordBoosting', exist_ok=True)
    os.makedirs('/tmp/Chunking', exist_ok=True)
    os.makedirs('/tmp/models/models--jhgan--ko-sroberta-sts/snapshots/3efa8e54a06798b00bd1abb9c22b2dd530e22b24/', exist_ok=True)

    os.makedirs("/tmp/.cache/huggingface", exist_ok=True)
    os.makedirs('/tmp/matplotlib', exist_ok=True)
    os.makedirs('/tmp/numba_cache', exist_ok=True)

    input_domain = event.get('domain', 'IT')
    mp3_file_url = event.get('mp3FileUrl', "input_audio.mp3")

    input_audio_file = '/tmp/STT/stt_audio/input_audio.mp3'
    output_txt_file = '/tmp/STT/stt_text/stt_text.txt'
    output_chunk_dict = '/tmp/Chunking/chunking_text.json'

    download_from_s3(mp3_file_url, input_audio_file)

    keyword_boosting_domain = f'STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json'
    keyword_boosting_agenda = 'STT/stt_text/KeywordBoosting/Agenda_middle.json'
    local_keyword_boosting_domain = f'/tmp/STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json'
    local_keyword_boosting_agenda = '/tmp/STT/stt_text/KeywordBoosting/Agenda_middle.json'

    download_from_s3(keyword_boosting_domain, local_keyword_boosting_domain)
    download_from_s3(keyword_boosting_agenda, local_keyword_boosting_agenda)

    local_model_dir = '/tmp/models'
    os.makedirs(local_model_dir, exist_ok=True)

    model_folder = 'models/models--jhgan--ko-sroberta-sts/'

    for file in model_folder:
        s3_model_folder = f'models/{file}'
        local_model_folder = os.path.join(local_model_dir, file)
        print(f"Downloading model folder from S3: {s3_model_folder}")
        download_folder_from_s3(s3_model_folder, local_model_folder)

    make_stt_txt(
        input_domain,
        input_audio_file,
        output_txt_file,
    )
    print(f"STT 파일 생성 완료 : {output_txt_file}")

    make_chunk(output_txt_file, output_chunk_dict)
    print(f"Chunk Dict 파일 생성 완료 : {output_chunk_dict}")

    upload_to_s3(output_txt_file, 'STT/stt_text/stt_text.txt')
    upload_to_s3(output_chunk_dict, 'Chunking/chunking_text.json')

    sagemaker_client = boto3.client('sagemaker', region_name='eu-north-1')

    processing_job_name = f"generate-summary-{context.aws_request_id}"

    response = sagemaker_client.start_processing_job(
        ProcessingJobName=processing_job_name,
        ProcessingInputs=[
            {
                'InputName': 'input-data',
                'S3Input': {
                    'S3Uri': f's3://clerkerai/Chunking/chunking_text.json',
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File'
                }
            },

            {
                'InputName': 'model-data',
                'S3Input': {
                    'S3Uri': f's3://clerkerai/models/models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/',
                    'LocalPath': '/opt/ml/processing/models',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File'
                }
            },

            {
                'InputName': 'font-data',
                'S3Input': {
                    'S3Uri': f's3://clerkerai/Keywords/NanumFontSetup_TTF_SQUARE_ROUND/NanumSquareRoundB.ttf',
                    'LocalPath': '/opt/ml/processing/fonts',
                    'S3DataType': 'S3Object',
                    'S3InputMode': 'File'
                }
            },
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'output-data',
                    'S3Output': {
                        'S3Uri': f's3://clerkerai/ProcessingOutput/',
                        'LocalPath': '/opt/ml/processing/output',
                        'S3UploadMode': 'EndOfJob'
                    }
                },
            ]
        },
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.t3.medium', #ml.g4dn.xlarge
                'VolumeSizeInGB': 10
            }
        },
        AppSpecification={
            'ImageUri': '715841353635.dkr.ecr.eu-north-1.amazonaws.com/clerker-ai:latest',
            'ContainerEntrypoint': ['python3', 'sagemaker_entrypoint.py'],
            'ContainerArguments': [
                '--input_chunk_dict', '/opt/ml/processing/input/chunking_text.json',
                '--diagram_summary_json', '/opt/ml/processing/output/diagram_summary.json',
                '--report_summary_json', '/opt/ml/processing/output/summary.json',
                '--model_id', '/opt/ml/processing/models/',
                '--model_path', '/opt/ml/processing/models/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
                '--font_path', '/opt/ml/processing/fonts/NanumSquareRoundB.ttf'
            ]
        },
        RoleArn='arn:aws:iam::your-account-id:role/your-sagemaker-execution-role'
    )

    print(f"SageMaker Processing Job '{processing_job_name}'이 시작되었습니다.")

    response = {
        "message": f"SageMaker Processing Job '{processing_job_name}' started.",
        "processing_job_name": processing_job_name
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }