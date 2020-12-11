import boto3
from FaceAPI import credentials


# sample path: 'dataset2/hinge_loss/center/amp/center_matt_03_leo_0.0_hing_amp_0.0.png'
def aws_celebrity(path_to_file):

    client = boto3.client('rekognition',
                          aws_access_key_id = credentials.aws_access_key_id,
                          aws_secret_access_key= credentials.aws_secret_access_key,
                          region_name = credentials.aws_region)

    if str.startswith(path_to_file, 'https://s3.amazonaws.com/797qjz1donyyji5r4n'):
        path_to_file = path_to_file[44:]
        # Update the new path_to_file
        # print(path_to_file)
    response = client.recognize_celebrities(
        Image={
            'S3Object': {
                'Bucket': credentials.s3_bucket_name,
                'Name': path_to_file
            }
        }
    )
    return response


def aws_compare(path1, path2, threshold, cred):
    #print('calling aws_compare')
    if cred == '0':
        aws_key = credentials.aws_access_key_id
        aws_secret = credentials.aws_secret_access_key
    elif cred == '1':
        aws_key = credentials.aws_access_key_id_new
        aws_secret = credentials.aws_secret_access_key_new
    else:
        aws_key = credentials.aws_access_key_id
        aws_secret = credentials.aws_secret_access_key
    client = boto3.client('rekognition',
                          aws_access_key_id=aws_key,
                          aws_secret_access_key=aws_secret,
                          region_name=credentials.aws_region)

    if str.startswith(path1, 'https://s3.amazonaws.com/797qjz1donyyji5r4n'):
        path1 = path1.replace('https://s3.amazonaws.com/797qjz1donyyji5r4n/', '')
        # Update the new path_to_file
        #print(path1)
    if str.startswith(path2, 'https://s3.amazonaws.com/797qjz1donyyji5r4n'):
        path2 = path2.replace('https://s3.amazonaws.com/797qjz1donyyji5r4n/', '')
        # Update the new path_to_file
    # print(path1)
    # print(path2)
    response = client.compare_faces(
        SourceImage={
            'S3Object': {
                'Bucket': credentials.s3_bucket_name,
                'Name': path1
            }
        },
        TargetImage={
            'S3Object': {
                'Bucket': credentials.s3_bucket_name,
                'Name': path2
            }
        },
        SimilarityThreshold=threshold
    )
    #print('response is:')
    #print(response) # For debug information

    matching_face_count = len(response['FaceMatches'])
    unmatched_face_count = len(response['UnmatchedFaces'])
    #print("Found {} matching faces".format(str(matching_face_count)))
    #print("Found {} unmatched faces".format(str(unmatched_face_count)))
    if matching_face_count != 0:
        return response['FaceMatches'][0]['Similarity']
    else:
        return None
    
