import os

subs = ['Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'JacksonCore','JacksonXml','Jsoup', 'JxPath']
versions = {'Cli':list(range(1, 6)) + list(range(7, 41)), 'Codec':range(1, 19), 'Collections': range(25, 29) , 'Compress': range(1, 48), 'Csv':range(1, 17), 'JacksonCore': range(1, 27),'JacksonXml':range(1, 7),'Jsoup':range(54, 94), 'JxPath': range(1, 23)}

for sub in subs:
	print(sub)
	for i in versions[sub]:
		print(i)
		cmd = 'cp -rf /raid/zqh/d4j/' + sub + str(i) + '/target/ /raid/zqh/FLocalization/tran_data/compile_files/' + sub + str(i)
		os.system(cmd)  
