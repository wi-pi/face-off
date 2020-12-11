models = ['triplet', 'center', 'large']
losses = ['hinge', 'target']
dataset_nums = ['1', '2', '3', '4']
test_nums = ['1','2']
apis = ['awsverify', 'azure', 'facepp']


for api in apis:
    print "#"+api
    for model in models:
        for loss in losses:
            for dataset_num in dataset_nums:
                if dataset_num != '3':
                    attack = 'cw'
                else:
                    attack = 'pgd'
                for test_num in test_nums:
                    print "python plot_results.py "+model+" "+loss+" "+dataset_num+" "+test_num+" "+api+" "+attack+" >> ./logs/" +api+"_out.log 2>> ./logs/" + api+"_err.log"
    print "\n"
