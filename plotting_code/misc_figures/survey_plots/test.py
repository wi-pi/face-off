f1 = open('test.csv','r')
f2 = open('test1.csv','w')
for line in f1:
	s = ''
	#print line.strip('\n').split('\t')
	for item in line.strip('\n').split('\t'):
		x = item.split('/')[1]
                y = ((float(x) - 1.8)*1000)-480
		s += str(1.2+(y* 0.01))+','
	s += '\n'
	f2.write(s)
f2.close()	
