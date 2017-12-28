def posi2pdb(pdb_filename, posi_filename):
    file = open(pdb_filename, 'w')
    submatrix = []
    with open(posi_filename, 'r') as file1:
        count = 0
        # generate "ATOM" part
        for line in file1:
            count += 1
            parse = line.strip().split()
            str1 = round(float(parse[0])+2, 3) 
            str2 = round(float(parse[1])+2, 3)
            str3 = round(float(parse[2])+2, 3) 
            print ("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
                   .format('ATOM', count, 'CA', ' ', 'MET', 'A', count, ' ', str1, str2, str3, 0.20, 10.00, ' ', ' '), file=file)
        # generate "CONNECT" part 
        for idx in range(1, count):
            print('CONNECT\t' + str(idx) + '\t' + str(idx+1))
        
        print('END')
    file.close()