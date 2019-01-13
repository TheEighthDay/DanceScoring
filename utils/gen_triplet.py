import numpy as np
import os
import random

train_rate=0.8
validation_rate=0.9
test_rate=1.0


people=['data/image/1/a','data/image/1/b','data/image/2/c','data/image/2/d','data/image/2/e','data/image/3/f',
'data/image/3/g','data/image/3/h','data/image/4/i','data/image/4/j','data/image/4/k','data/image/5/l','data/image/5/m',
'data/image/6/n','data/image/6/o','data/image/7/p','data/image/7/q','data/image/7/r','data/image/8/s','data/image/8/t',
'data/image/8/u','data/image/9/v','data/image/9/w','data/image/9/x','data/image/10/y','data/image/10/z',
'data/image/11/aa','data/image/11/bb']
#每个人舞姿的文件夹。

video_seconds=[103,216,78,224,119,19,120,177,127,198,258]
#每个人视频总共的秒数。

video2people=[[0,1],[2,3,4],[5,6,7],[8,9,10],[11,12],[13,14],[15,16,17],[18,19,20],[21,22,23],[24,25],[26,27]]
#视频对应人

def write_txt(path,data):
    with open("../data/txt/"+path,'w') as f:
        for x in data:
            for y in x:
                f.write(y)
                f.write(',')
            f.write('\n')
        f.close()





def gen_triplet_two_people(video_id):
    video=[i for i in range(0,video_seconds[video_id])]
    random.shuffle(video)

    train=[]
    validation=[]
    test=[]

    train_video=video[:int(video_seconds[video_id]*train_rate)]
    validation_video=video[int(video_seconds[video_id]*train_rate):int(video_seconds[video_id]*validation_rate)]
    test_video=video[int(video_seconds[video_id]*validation_rate):int(video_seconds[video_id]*test_rate)]

    for x in train_video:
        people1_second=[]
        for i in range(10):
            people1_second.append(people[video2people[video_id][0]]+'/'+str(x)+'_'+str(i)+'.jpg')
        people2_second=[]
        for i in range(10):
            people2_second.append(people[video2people[video_id][1]]+'/'+str(x)+'_'+str(i)+'.jpg')

        people3_second=[]
        vi=int(random.random()*len(video2people))
        peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
        sec=int(video_seconds[vi]*random.random())

        while(True):
            if not (sec==x and vi==video_id):
                break
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

        for i in range(10):
            people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

        people1_second.extend(people2_second)
        people1_second.extend(people3_second)
        train.append(people1_second)



    for x in validation_video:
        people1_second=[]
        for i in range(10):
            people1_second.append(people[video2people[video_id][0]]+'/'+str(x)+'_'+str(i)+'.jpg')
        people2_second=[]
        for i in range(10):
            people2_second.append(people[video2people[video_id][1]]+'/'+str(x)+'_'+str(i)+'.jpg')

        people3_second=[]
        vi=int(random.random()*len(video2people))
        peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
        sec=int(video_seconds[vi]*random.random())

        while(True):
            if not (sec==x and vi==video_id):
                break
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

        for i in range(10):
            people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

        people1_second.extend(people2_second)
        people1_second.extend(people3_second)
        validation.append(people1_second)

    for x in test_video:
        people1_second=[]
        for i in range(10):
            people1_second.append(people[video2people[video_id][0]]+'/'+str(x)+'_'+str(i)+'.jpg')
        people2_second=[]
        for i in range(10):
            people2_second.append(people[video2people[video_id][1]]+'/'+str(x)+'_'+str(i)+'.jpg')

        people3_second=[]
        vi=int(random.random()*len(video2people))
        peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
        sec=int(video_seconds[vi]*random.random())

        while(True):
            if not (sec==x and vi==video_id):
                break
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

        for i in range(10):
            people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

        people1_second.extend(people2_second)
        people1_second.extend(people3_second)
        test.append(people1_second)


    return train,validation,test

def gen_triplet_three_people(video_id):
    video=[i for i in range(0,video_seconds[video_id])]
    random.shuffle(video)

    train=[]
    validation=[]
    test=[]

    train_video=video[:int(video_seconds[video_id]*train_rate)]
    validation_video=video[int(video_seconds[video_id]*train_rate):int(video_seconds[video_id]*validation_rate)]
    test_video=video[int(video_seconds[video_id]*validation_rate):int(video_seconds[video_id]*test_rate)]

    dx=[0,1,2]
    dy=[1,2,0]
    for d in range(3):
        for x in train_video:
            people1_second=[]
            for i in range(10):
                people1_second.append(people[video2people[video_id][dx[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')
            people2_second=[]
            for i in range(10):
                people2_second.append(people[video2people[video_id][dy[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')

            people3_second=[]   #随机选择第三张图
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

            while(True):
                if not (sec==x and vi==video_id):
                    break
                vi=int(random.random()*len(video2people))
                peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
                sec=int(video_seconds[vi]*random.random())

            for i in range(10):
                people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

            people1_second.extend(people2_second)
            people1_second.extend(people3_second)
            train.append(people1_second)



        for x in validation_video:
            people1_second=[]
            for i in range(10):
                people1_second.append(people[video2people[video_id][dx[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')
            people2_second=[]
            for i in range(10):
                people2_second.append(people[video2people[video_id][dy[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')

            people3_second=[]
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

            while(True):
                if not (sec==x and vi==video_id):
                    break
                vi=int(random.random()*len(video2people))
                peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
                sec=int(video_seconds[vi]*random.random())

            for i in range(10):
                people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

            people1_second.extend(people2_second)
            people1_second.extend(people3_second)
            validation.append(people1_second)

        for x in test_video:
            people1_second=[]
            for i in range(10):
                people1_second.append(people[video2people[video_id][dx[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')
            people2_second=[]
            for i in range(10):
                people2_second.append(people[video2people[video_id][dy[d]]]+'/'+str(x)+'_'+str(i)+'.jpg')

            people3_second=[]
            vi=int(random.random()*len(video2people))
            peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
            sec=int(video_seconds[vi]*random.random())

            while(True):
                if not (sec==x and vi==video_id):
                    break
                vi=int(random.random()*len(video2people))
                peo=video2people[vi][0]+int(len(video2people[vi])*random.random())
                sec=int(video_seconds[vi]*random.random())

            for i in range(10):
                people3_second.append(people[peo]+'/'+str(sec)+'_'+str(i)+'.jpg')

            people1_second.extend(people2_second)
            people1_second.extend(people3_second)
            test.append(people1_second)


    return train,validation,test




def main():
    
    train=[]
    validation=[]
    test=[]
    for i in range(len(video2people)):
        l=len(video2people[i])
        if l==2:
            tr,va,te=gen_triplet_two_people(i)
            train.extend(tr)
            validation.extend(va)
            test.extend(te)
        else:
            tr,va,te=gen_triplet_three_people(i)
            train.extend(tr)
            validation.extend(va)
            test.extend(te)

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)

    print(len(train))
    print(len(validation))
    print(len(test))
    
    write_txt("train_2.txt",train)
    write_txt("validation_2.txt",validation)
    write_txt("test_2.txt",test)



if __name__ == '__main__':
    main()



        



