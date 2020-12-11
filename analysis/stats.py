import sys
import os
import numpy as np

ROOT = os.getcwd()

def per_src_target_pair():

    filename = sys.argv[1]
    input_amp = sys.argv[2]
    input_margin = sys.argv[3]


    if input_amp == 'all' and input_margin != 'all':
        amp_list = [1,2,3,4,5]
        margin_list = [int(input_margin)]
        CONST = 25
    elif input_margin == 'all' and input_amp != 'all':
        margin_list = [0.0,5.0,10.0]
        amp_list = [int(input_amp)]
        CONST = 15
    elif input_margin == 'all' and input_amp == 'all':
        margin_list = [0.0,5.0,10.0]
        amp_list = [1,2,3,4,5]
        CONST = 75
    else:
        amp_list = [int(input_amp)]
        margin_list = [int(input_margin)]
        CONST = 5

    #file_name = 'large_triplet-large_center_full.csv'

    file_name = filename

    f = open(os.path.join(ROOT, file_name), 'r')
    lines = f.readlines()
    lines = lines[1:-1]

    keys = {'barack':0, 'bill':1, 'jenn':2, 'leo':3, 'mark':4, 'matt':5, 'melania':6, 'meryl':7, 'morgan':8, 'taylor':9}
    d = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for line in lines:
        split = line.split(',')
        #print(split)
        margin = round(float(split[9]))
        #margin = round(float(split[11]))
        amp = round(float(split[10]))
        #amp = round(float(split[12]))
        src = split[4]
        target = split[5]
        match_src = split[6]
        match_target = split[7]
        if match_src.lower() == 'true':
            match_src = 1
        else:
            match_src = 0

        if match_target.lower() == 'true':
            match_target = 1
        else:
            match_target = 0

        cur = d[keys[src]]

        if amp in amp_list and margin in margin_list:
            if target not in cur:
                cur[target] = [match_src, match_target]
            else:
                tmp = cur[target]
                tmp[0] += match_src
                tmp[1] += match_target
                cur[target] = tmp

    def best_target(inp_list):
        max_val = 0
        best = None
        for i, item in enumerate(inp_list):
            elements = item.strip().split(',')
            target = elements[1]
            match_target = float(elements[5])
            if match_target >= max_val:
                max_val = match_target
                best = target
        print("Best target:", best)

    k = keys.keys()
    for item in k:
        temp = []
        cur_dict = d[keys[item]]
        dict_keys = cur_dict.keys()
        if len(dict_keys) >= 1:
            print("src:", item)
            for dict_key in dict_keys:
                string = "target," + str(dict_key) + ",match_src," + str(round(float(cur_dict[dict_key][0])/CONST*100,2)) + ",match_target," + str(round(float(cur_dict[dict_key][1])/CONST*100,2))
                print(string)
                temp.append(string)
            best_target(temp)
            print("\n")

def aggregate(filename, attack_type):
    #file_name = 'large_triplet-large_center.csv'
    file_name = os.path.join(ROOT, os.path.join('csv', os.path.join(attack_type, filename)))
    '''
    #margin_list = [0,5,10,11,12]
    margin_list = [10,11,12]
    #amp_list = [1,2,3,4,5]
    amp_list = list(np.arange(1,5,0.2))
    '''
    f = open(file_name, 'r')
    lines = f.readlines()
    lines = lines[1:-1]

    write_path = os.path.join(ROOT, os.path.join('txt', attack_type))
    if os.path.exists(write_path) == False:
        os.mkdir(write_path)
    write_path = os.path.join(write_path, filename.replace('.csv', '-aggregate.txt'))
    w = open(write_path, 'w')
    amp_list = set()
    margin_list = set()
    for line in lines:
        elements = line.strip().split(',')
        margin_list.add(elements[11])
        amp_list.add(elements[12])

    margin_list = list(margin_list)
    amp_list = list(amp_list)
    
    def convert_to_list(inp):
        for i, element in enumerate(inp):
            if '.' in element:
                inp[i] = float(element)
            else:
                inp[i] = int(element)

        return inp

    margin_list = convert_to_list(margin_list)
    amp_list = convert_to_list(amp_list)

    amp_list.sort(reverse=True)
    margin_list.sort(reverse=True)

    keys = {'barack':0, 'bill':1, 'jenn':2, 'leo':3, 'mark':4, 'matt':5, 'melania':6, 'meryl':7, 'morgan':8, 'taylor':9}
    d = {}
    n = len(lines)
    for line in lines:
        split = line.strip().split(',')
        margin = round(float(split[11]),2)
        amp = round(float(split[12]),2)
        key = str(margin) + '_' + str(amp)
        src = split[4]
        target = split[7]
        match_src = split[5]
        match_target = split[6]
        
        if match_src.lower() == 'true':
            match_src = 1
        else:
            match_src = 0
        
        if match_target.lower() == 'true':
            match_target = 1
        else:
            match_target = 0

        if key not in d:
            d[key] = [match_src, match_target, 1]
        else:
            tmp = d[key]
            tmp[0] += match_src
            tmp[1] += match_target
            tmp[2] += 1
            d[key] = tmp

    final_output = []
    for amp in amp_list:
        for margin in margin_list:
            margin = round(margin, 2)
            amp = round(amp, 2)
            key = str(margin) + '_' + str(amp)
            n = d[key][2]
            match_src = round(float(d[key][0])/n * 100, 2)
            match_target = round(float(d[key][1])/n * 100, 2)
            string = "amp,"+str(amp)+",margin,"+str(margin)+",match_src,"+str(match_src)+",match_target,"+str(match_target)
            #print(string)
            w.write(string+"\n")
            final_output.append(string)

    max_val = 0
    N = len(amp_list)
    min_amp = amp_list[N-1]
    index = 0
    for i, output in enumerate(final_output):
        elements = output.strip().split(',')
        amp_val = float(elements[1])
        margin_val = float(elements[3])
        match_src_val = float(elements[5])
        match_target_val = float(elements[7])
        if match_target_val >= max_val:
            #if amp_val <= min_amp:
            #print(elements)
            max_val = match_target_val
            min_amp = amp_val
            index = i

    #print("\nBEST")
    w.write("\nBEST \n")
    #print(final_output[index])
    w.write(final_output[index]+"\n")
    w.close()

if __name__ == "__main__":
    
    aggregate(sys.argv[1], sys.argv[2])
    #per_src_target_pair() #usage: python stats.py filename all all
