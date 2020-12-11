import requests
import Config
from FaceAPI import credentials


def microsoft_azure_face(photo_binary_list, cred):
    # Microsoft Azure Face Detection:
    # In order to use Verify function to check 2 faces, need to call detect first to get FaceID
    if cred == '0':
        key = credentials.FaceID_key
    elif cred == '1':
        key = credentials.azure_key
    else:
        key = credentials.FaceID_key

    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': key}
    query_params = {'returnFaceId': 'true'} # Required by verify to pass as parameter
    detected_faceId = []
    try:
        for photo in photo_binary_list:
            r = requests.post(credentials.FaceID_endpoint + '/detect', params=query_params, data=photo, headers=headers)
            face_list = r.json()
            if len(face_list) != 1:
                print('Error: Microsoft Azure Face found this number of faces in the image: ' + str(len(face_list)))
                print("Operation Aborted")
                return None
            #print(face_list) # Provide debugging info

            # There should only be 1 face in each image
            face_id = face_list[0]['faceId']
            detected_faceId.append(face_id)
                
        # Now perform the verify
        headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': key}
        payload = {'faceId1': detected_faceId[0], 'faceId2': detected_faceId[1]}
        r = requests.post(credentials.FaceID_endpoint + '/verify', json=payload, headers=headers)
        return r.json()
    except Exception as e:
        print(e)
        return None


def verify_API(photo1, photo2, service, cred):
    local_photo1 = photo1.replace(Config.S3_DIR, Config.ROOT)
    local_photo2 = photo2.replace(Config.S3_DIR, Config.ROOT)
    photo_binary_list = []
    photo_binary_list.append(open(local_photo1, "rb").read())
    photo_binary_list.append(open(local_photo2, "rb").read())

    if service is 1:
        result =  microsoft_azure_face(photo_binary_list, cred)
        #print(result)
        if result is None or 'confidence' not in result:
            return None, None, False
        else:
            return result['confidence'], result['isIdentical'], True
        
