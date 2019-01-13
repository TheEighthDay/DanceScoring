
import json
import cv2
import numpy as np
from utils_json import read_json
from resize_300 import let_people_full_of_screen
from tqdm import tqdm

def video2image(fps,number,first,second):
    #fps视频每秒帧数
    #number 抽取帧数的间隔
    #first 第几个视频
    #second 该共几个片段（每个视频被划分成30秒的n个片段）
    for k in range(1,second+1):
        cap = cv2.VideoCapture('../data/video_match/'+str(first)+'_'+str(k)+'.avi')
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
            break

        i=0
        j=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if i%number==0:
                    cv2.imwrite("../data/image/"+str(first)+'/'+str(k)+"_"+str(j)+"_"+str(int(i/number))+".jpg",frame)
            else:
                print("over")
                break
            i+=1
            i=i%fps
            if(i==0):
                j+=1
        cap.release()

def location(people):
    keypoints=people['keypoints']
    l=[]
    for i in range(len(keypoints)): #什么含义？
        if i%3==0:
            l.append(keypoints[i])
    l=np.array(l)
    return [l.max(),l.min()]







def video2image_all_two_people(fps,number,first,second):#fps,number,
    #fps视频每秒帧数
    #number 抽取帧数的间隔
    #first 第几个视频
    cap = cv2.VideoCapture('../data/video_match/'+str(first)+'.mp4')
    data = read_json(str(first)+'.json')
    data_number = [int(data[i]['image_id'].split('.')[-2]) for i in range(len(data))]
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    k=0 #删除噪音的帧
    j=0 #秒
    all_frame=int(data[-1]['image_id'].split('.')[-2])
    for i in tqdm(range(all_frame+1)):
        if data_number.count(i)==2:
            ret,frame = cap.read()
            if (k%number==0):
                people1_index=data_number.index(i)
                people2_index=data_number.index(i)+1

                people1=data[people1_index]
                people2=data[people2_index]

                people1_location=location(people1)
                people2_location=location(people2)

                # print(k,people1_location,people2_location)

                people1_location.extend(people2_location)
                loaction=np.sort(people1_location)

                mid=int((loaction[1]+loaction[2])//2)
                try:
                    frame1=let_people_full_of_screen(frame[:,:mid,:])  
                    frame2=let_people_full_of_screen(frame[:,mid:,:]) 


                    cv2.imwrite("../data/image/"+str(first)+'/'+second[0]+'/'+str(j)+"_"+str(int(k/number))+".jpg",frame1)
                    cv2.imwrite("../data/image/"+str(first)+'/'+second[1]+'/'+str(j)+"_"+str(int(k/number))+".jpg",frame2)
                except:
                    print("error+1")
                #制作图片


            k+=1
            k=k%fps
            if(k==0):
                j+=1

        else:
            ret,frame = cap.read()


def video2image_all_three_people(fps,number,first,second):#fps,number,
    #fps视频每秒帧数
    #number 抽取帧数的间隔
    #first 第几个视频
    cap = cv2.VideoCapture('../data/video_match/'+str(first)+'.mp4')
    data = read_json(str(first)+'.json')
    data_number = [int(data[i]['image_id'].split('.')[-2]) for i in range(len(data))]
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    k=0 #删除噪音的帧
    j=0 #秒
    all_frame=int(data[-1]['image_id'].split('.')[-2])
    for i in tqdm(range(all_frame+1)):
        if data_number.count(i)==3:
            ret,frame = cap.read()
            if (k%number==0):
                people1_index=data_number.index(i)
                people2_index=data_number.index(i)+1
                people3_index=data_number.index(i)+2

                people1=data[people1_index]
                people2=data[people2_index]
                people3=data[people3_index]

                people1_location=location(people1)
                people2_location=location(people2)
                people3_location=location(people3)

                # print(k,people1_location,people2_location)

                people1_location.extend(people2_location)
                people1_location.extend(people3_location)
                loaction=np.sort(people1_location)

                mid1=int((loaction[1]+loaction[2])//2)
                mid2=int((loaction[3]+loaction[4])//2)

                frame1=let_people_full_of_screen(frame[:,:mid1,:]) 
                frame2=let_people_full_of_screen(frame[:,mid1:mid2,:]) 
                frame3=let_people_full_of_screen(frame[:,mid2:,:]) 


                cv2.imwrite("../data/image/"+str(first)+'/'+second[0]+'/'+str(j)+"_"+str(int(k/number))+".jpg",frame1)
                cv2.imwrite("../data/image/"+str(first)+'/'+second[1]+'/'+str(j)+"_"+str(int(k/number))+".jpg",frame2)
                cv2.imwrite("../data/image/"+str(first)+'/'+second[2]+'/'+str(j)+"_"+str(int(k/number))+".jpg",frame3)

                #制作图片


            k+=1
            k=k%fps
            if(k==0):
                j+=1

        else:
            ret,frame = cap.read()



    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         if judge_no_noise(i):
    #             if i%number==0:
    #                 #制作图片
    #             i+=1
    #             i=i%fps
    #             if(i==0):
    #                 j+=1


    #     else:
    #         print("the video over")
    #         break;

    # cap.release()



    # people1 = [data[i] for i in range(len(data)) if i%2==0]
    # people2 = [data[i] for i in range(len(data)) if i%2==1]
    # people1 = [people1[i]['image_id'].split('.')[-2] for i in range(len(people1))]
    # people2 = [people2[i]['image_id'].split('.')[-2] for i in range(len(people2))]
    # print(len(people1),len(people2),len(data),people1,people2)
    # i=0
    # j=0
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         if i%number==0:
    #             cv2.imwrite("../data/image/"+str(first)+'/'+str(j)+"_"+str(int(i/number))+".jpg",frame)
    #     else:
    #         print("over")
    #         break
    #     i+=1
    #     i=i%fps
    #     if(i==0):
    #         j+=1
    # cap.release()



def count_frame(video): #3252
    count=0
    cap = cv2.VideoCapture('../data/video_match/'+video)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret==True):
            count+=1
            print(count)
    return count





def main():
    # print(count_frame("1.mp4"))
    # video2image_all_three_people(30,3,4)
    # video2image_all_two_people(30,3,,["n","o"])
    
    video2image_all_two_people(30,3,11,["aa","bb"])

if __name__ == '__main__':
    main()
