import numpy as np

def generate(feature=[], prob=0.9):
    f = open("person_0_0.csv", "r")
    lines = f.readlines()
    write_f = open('node_table', 'w')
    write_n = open('node_map', 'w')
    lines_to_write = ['id:int64\tfeature:string\n']
    map_to_write = ['id:int64\tseq:int64\n']
    for i in range(1, len(lines)):
        l = lines[i].split("|")
        if len(feature) == 0:
            line = l[0]+'\t'+str(0)+'\n'
        else:
            line = l[0]+'\t'
            for idx, j in enumerate(feature):
                if idx == 0:
                    import numpy as np
                    if j == 4: #birth date
                        num_date = (float(l[j]) - 19000000.0)
                        num_date = num_date + np.random.uniform(0, 10000.0)
                        on = np.random.uniform(0, 100.0)
                        fj = str(num_date) + ":" + str(on)
                    line += fj
                else:
                    line += (':' + l[j][:-1])
            line += '\n'
        map_line = l[0] + '\t' + str(i - 1) + '\n'
        lines_to_write.append(line)
        map_to_write.append(map_line)
    write_f.writelines(lines_to_write)
    write_n.writelines(map_to_write)
    write_f.close()
    write_n.close()
    f.close()
    
    import numpy as np
    f = open("person_knows_person_0_0.csv", "r")
    lines = f.readlines()
    write_train_f = open('edge_table_train', 'w')
    write_test_f = open('edge_table_test', 'w')
    train_to_write = ['src_id:int64\tdst_id:int64\tweight:double\n']
    test_to_write = ['src_id:int64\tdst_id:int64\tweight:double\n']
    for i in range(1, len(lines)):
        a = np.random.choice(2, p=[prob, 1-prob])
        l = lines[i].split("|")
        line = l[0]+'\t'+l[1]+'\t'+'1.0'+'\n'
        if a == 0:
            train_to_write.append(line)
        else:
            test_to_write.append(line)
    write_train_f.writelines(train_to_write) 
    write_train_f.close() 
    write_test_f.writelines(test_to_write) 
    write_test_f.close() 

if __name__ == "__main__":
    feature=[4, 7]
    generate(feature, prob=0.5)
