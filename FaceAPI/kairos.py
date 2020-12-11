import requests
from FaceAPI import credentials
import json



def kairos_enroll_to_gallery(image, subject_id):
    url = credentials.kairos_url + '/enroll'
    headers = {
        "app_id": credentials.kairos_app_id,
        "app_key": credentials.kairos_key
    }

    payload = {
        "image": image,
        "subject_id": subject_id,
        "gallery_name": credentials.kairos_gallery_name
    }

    r = requests.post(url, data=json.dumps(payload), headers=headers)
    print(r.text)


def kairos_verify(photo, subject_id):
    url = credentials.kairos_url + '/verify'

    headers = {
        "app_id": credentials.kairos_app_id,
        "app_key": credentials.kairos_key
    }

    payload = {
        "image": photo,
        "gallery_name": credentials.kairos_gallery_name,
        "subject_id": subject_id
    }

    r = requests.post(url, data=json.dumps(payload), headers=headers)

    result = r.json()
    if 'images' not in result:
        print(result)
    else:
        for image in result['images']:
            print(image['transaction']['confidence'])


