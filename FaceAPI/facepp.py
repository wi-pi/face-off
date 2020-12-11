from FaceAPI import credentials
import requests

# If the confidence does not meet the "1e-3" threshold, 
# it is highly suggested that the two faces are not from the same person. 
# While if the confidence is beyond the "1e-5" threshold, 
# there's high possibility that they are from the same person.
# 1e-3: confidence threshold at the 0.1% error rate;
# 1e-4: confidence threshold at the 0.01% error rate;
# 1e-5: confidence threshold at the 0.001% error rate;
def facepp(face1, face2, cred):
    if cred == '0':
        key = credentials.facepp_api_key
        secret = credentials.facepp_api_sec
    elif cred == '1':
        key = credentials.facepp_api_key_new
        secret = credentials.facepp_api_sec_new
    else:
        key = credentials.facepp_api_key
        secret = credentials.facepp_api_sec
    url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
    payload = {
        'api_key': key,
        'api_secret': secret,
        'image_url1': face1,
        'image_url2': face2
    }
    error = True
    while error:
        error = False
        r = requests.post(url, data=payload)
        response = r.json()
        if 'error_message' in response and response['error_message'] == 'CONCURRENCY_LIMIT_EXCEEDED':
            error = True
        # print('==================')
        # print(face1)
        # print(r.text)
    try:
        score = response['confidence']
        th = [response['thresholds']['1e-3'], response['thresholds']['1e-4'], response['thresholds']['1e-5']]
    except Exception as e:
        print(response)
        score = None
        th = None
        print(e)
    return score, th
