apis = ['facepp', 'azure', 'awsverify']
attacks = ['cw_l2', 'cw_li', 'pgd_l2']
models = ['large_triplet', 'small_center', 'large_center', 'casia', 'small_triplet']

for api in apis:
    print("#"+api)
    for attack in attacks:
        for model in models:
            print("python read_and_plot.py "+api+" "+attack+" "+model+" >> "+api+"_table.txt")
    print("\n")


