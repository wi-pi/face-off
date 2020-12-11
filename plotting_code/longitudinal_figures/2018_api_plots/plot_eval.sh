#azure
python plot_results.py triplet target 2 1 azure cw >> api_out.log 2>> api_err.log
python plot_results.py triplet target 2 2 azure cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 1 azure cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 2 azure cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 1 azure cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 2 azure cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 1 azure cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 2 azure cw >> api_out.log 2>> api_err.log

#awsverify
python plot_results.py triplet target 2 1 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py triplet target 2 2 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 1 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 2 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 1 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 2 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 1 awsverify cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 2 awsverify cw >> api_out.log 2>> api_err.log

#facepp
python plot_results.py triplet target 2 1 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py triplet target 2 2 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 1 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py triplet hinge 2 2 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 1 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py center target 2 2 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 1 facepp cw >> api_out.log 2>> api_err.log
python plot_results.py center hinge 2 2 facepp cw >> api_out.log 2>> api_err.log

#pgd
python plot_results.py center hinge 3 1 azure pgd >> api_out.log 2>> api_err.log
python plot_results.py center hinge 3 2 azure pgd >> api_out.log 2>> api_err.log
python plot_results.py center hinge 3 1 awsverify pgd >> api_out.log 2>> api_err.log
python plot_results.py center hinge 3 2 awsverify pgd >> api_out.log 2>> api_err.log
python plot_results.py center hinge 3 1 facepp pgd >> api_out.log 2>> api_err.log
python plot_results.py center hinge 3 2 facepp pgd >> api_out.log 2>> api_err.log

