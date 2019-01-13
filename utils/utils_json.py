import json


def read_json(path):
	with open("../data/json/"+path,'r') as load_f:
		load_dict = json.load(load_f)
	return load_dict

def combine(list_paths):
	ending=[]
	flag=read_json(list_paths[0])


	ending.extend(flag)
	last=flag[-1]
	last=int(last['image_id'].split('.')[-2])+1
	print(last)
	for i in range(1,len(list_paths)):
		flag=read_json(list_paths[i])
		for x in flag:
			x['image_id']=str(int(x['image_id'].split('.')[-2])+last)+'.jpg'
		last=flag[-1]
		last=int(last['image_id'].split('.')[-2])+1
		print(last)
		ending.extend(flag)
	return ending

def write_json(data,path):
	with open("../data/json/"+path,'w') as json_file:
	    json.dump(data,json_file)



if __name__ == '__main__':
    list_paths=[]
    # list_paths=["4_1.json","4_2.json","4_3.json","4_4.json","4_5.json","4_6.json","4_8.json","4_9.json"]
    for i in range (1,10):
        list_paths.append("11_"+str(i)+".json")
    ending=combine(list_paths)
    write_json(ending,"11.json")
    # a=read_json("5_1.json")
    # print(a)






         