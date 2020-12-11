import os

ROOT = os.getcwd()


def process(file_name, src_constraint, target_constraint):
    elems = file_name.split('-')
    t1 = elems[0].split('_')
    t2 = elems[1].split('_')
    t3 = elems[2].split('.')[0].split('_')
    attack = t3[0]
    norm = t3[1]
    src, target = t1[2], t2[0]

    constraints = src_constraint != None or target_constraint != None

    if constraints == False:
        return file_name
    else:
        if src_constraint != None:
            if src == src_constraint:
                if target_constraint != None:
                    if target == target_constraint:
                        return file_name
                    else:
                        return None
                else:
                    return None
            else:
                return None

def return_files(path, src_constraint='large', targ_constraint='large'):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    final = []
    for f in files:
        if 'csv' in f:
            val = process(f, src_constraint, targ_constraint)
            if val != None:
                final.append(val)

    return final


def check(src_class, desired_src, target_class, desired_target, amplification, desired_amp, margin, desired_margin):
    
    if desired_src and desired_target and desired_amp and desired_margin:
        if src_class == desired_class and target_class == desired_target and amplification == desired_amp and margin == desired_margin:
            return True
    else:


def plot(file_name, desired_src, desired_target, desired_amp, desired_margin):
    f = open(file_name, 'r')
    lines = f.readlines()
    header = lines[0]
    print(header)
    lines = lines[1:]
    for line in lines:
        elements = line.strip().split(',')
        #model_name,target_model_name,attack_name,attack_loss,source,target,match_source,match_target,image_name,margin,amplification,top1,distance1,top2,distance2,top3,distance3,top4,distance4,top5,distance5
        #large_triplet,large_center_vgg,cw_l2,hinge,taylor,morgan,False,True,taylor_03,10.00,5.000,morgan,11.140576,melania,12.363316,meryl,13.337789,barack,13.620404,bill,13.935996
        src_model = elements[0]
        target_model = elements[1]
        attack_norm = elements[2]
        attack_loss = elements[3]
        src_class = elements[4]
        target_class = elements[5]
        src_match = bool(elements[6])
        target_match = bool(elements[7])
        src_img = elements[8]
        margin = float(element[9])
        amplification = float(element[10])
        t = []
        d = []
        for i in range(0,5):
            index = 11+i
            t.append(element[index])
            d.append(float(element[index])


        ret_val = check(src_class, desired_src, target_class, desired_target, amplification, desired_amp, margin, desired_margin)
        if ret_val == True:
            
        else:
            continue

if __name__ == "__main__":
    data_src_path = os.path.join(ROOT, 'csv')
    files = return_files(data_src_path)
    for fil in files:
        fil_path = os.path.join(data_src_path, fil)
        plot(fil_path)
        exit()

