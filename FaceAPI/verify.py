from FaceAPI import microsoft_face
from FaceAPI import google_image
from FaceAPI import aws_rekognition
from FaceAPI import facepp
from FaceAPI import kairos
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("service", type=int, help="Select Facial Verification Service to use")
parser.add_argument("photo1", help="Path or URL to the first photo")
parser.add_argument("photo2", default=None, nargs='?', help="Path or URL to the second photo")
parser.add_argument("threshold", type=float, default=None, nargs='?', help="threshold value for aws matching, in 100 scale")

args = parser.parse_args()

args_photo_list = []
args_photo_list.append(args.photo1)
args_photo_list.append(args.photo2)

if args.service == 1:
    microsoft_face.microsoft_azure_face(args_photo_list)
elif args.service == 2:
    if args.photo2 is not None:
        print("Note: Google Image search will only return best guess for the photo1, photo2 is ignored.")
    google_image.google_image_search_by_url(args_photo_list[0])
elif args.service == 3:
    if args.photo2 is not None:
        print("Note: AWS Celebrity recognition will only return best guess for the photo1, photo2 is ignored.")
    aws_rekognition.aws_celebrity(args_photo_list[0])
elif args.service == 4:
    aws_rekognition.aws_compare(args_photo_list[0], args_photo_list[1], args.threshold)
elif args.service == 5:
    facepp.facepp(args_photo_list[0], args_photo_list[1])
elif args.service == 6:
    # Enroll the photo into gallery with subject_ID in the path
    kairos.kairos_enroll_to_gallery(args_photo_list[0], args_photo_list[1])
elif args.service == 7:
    # Enroll the photo into gallery with subject_ID in the path
    kairos.kairos_verify(args_photo_list[0], args_photo_list[1])

