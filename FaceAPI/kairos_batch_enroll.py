import os
from FaceAPI import credentials
from FaceAPI import kairos
import base64

s3_base = 'https://s3.amazonaws.com/797qjz1donyyji5r4n/aligned_imgs/'

count = 0

files = [f for f in os.listdir('.') if os.path.isfile(f)]

for root, dirs, files in os.walk('.'):
    for name in files:
        full_path = str(os.path.join(root, name))
        full_path = full_path.replace(' ', '+')
        full_path = full_path.replace('\\', "/")
        s3_path = s3_base + full_path[2:]
        print(s3_path)
        subject_id = os.path.basename(root)
        print(subject_id)
        kairos.kairos_enroll_to_gallery(s3_path, subject_id)



    # for name in dirs:
    #     print(os.path.join(root, name))

#
# for file_name in files:
#     print(os.path.basename(file_name))
#     if "png" in file_name:
#        # print(file_name)
#
#         count = count + 1
#
# print("Total upload: " + str(count))
