import os

ROOT = os.getcwd()

dataset_nums = ['1', '2', '3', '4']
apis = ['awsverify', 'azure', 'faceapp']

path = os.path.join(ROOT, 'api_results')

for api in apis:
    path1 = os.path.join(path, api)
    for dataset_num in dataset_nums:
        path_final = os.path.join(path1, 'dataset'+dataset_num)
        if os.path.isdir(path_final) == True:
            files = os.listdir(path_final)
            print(path_final)
            print(files)
            print("\n")
