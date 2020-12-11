# python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm 2 --targeted-flag true --tv-flag false \
# --hinge-flag true --cos-flag false --override-dir test_imgs

# python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm 2 --targeted-flag true --tv-flag false \
# --hinge-flag true --cos-flag false --override-dir celeb96

# python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm 2 --targeted-flag true --tv-flag false \
# --hinge-flag true --cos-flag false --override-dir celeb160

# python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm 2 --targeted-flag true --tv-flag false \
# --hinge-flag true --cos-flag false --override-dir celeb

python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag false

#python FaceAPI/s3_upload.py --model-type small --loss-type center --attack CW --norm 2 --targeted-flag true --tv-flag false \
#--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag false

#python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack CW --norm inf --targeted-flag true --tv-flag false \
#--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag true

#python FaceAPI/s3_upload.py --model-type small --loss-type center --attack CW --norm inf --targeted-flag true --tv-flag false \
#--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag true

#python FaceAPI/s3_upload.py --model-type large --loss-type triplet --attack PGD --norm 2 --targeted-flag true --tv-flag false \
#--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag true

#python FaceAPI/s3_upload.py --model-type small --loss-type center --attack PGD --norm 2 --targeted-flag true --tv-flag false \
#--hinge-flag true --cos-flag false --margin 15.0 --amplification 10.6 --granularity api-eval --mean-loss embedding --pair-flag true
