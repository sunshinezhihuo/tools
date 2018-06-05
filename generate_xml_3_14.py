#--coding:utf-8--
import os
from unicoe_tool.generate_xml import generate_xml

def folder_struct(level, path):
    global allFileNum

    dirList = []
    fileList = []
    files = os.listdir(path)
    dirList.append(str(level))

    for f in files:
        if(os.path.isdir(path + '/' + f)):
            if f[0] != '.':
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)

    i_dl = 0
    for dl in dirList:
        if i_dl == 0:
            i_dl = i_dl + 1
        else:
            #print '-' * (int(dirList[0])), dl
            folder_struct((int(dirList[0]) + 1), path+'/'+dl)

    # print fileList
    # print dirList
    cnt = 0
    wf = open("/home/user/PycharmProjects/caltech_new_anno/train1x.txt", "w")
    print len(fileList)
    for fl in fileList:
        #todo 遍历txt文件，然后根据txt文件生成xml文件
        tmp = path + "/" + fl

        rtmp = open(tmp, "r")
        # wf.write(fl)
        # wf.write("----")
        # wf.write('\n')
        contx = rtmp.readline()

        fileInfo = []
        obj = []
        cor_dict = {}
        flag = 0

        while contx:
            contx = rtmp.readline()
            if contx:

                if flag == 0:
                    fileInfo.append("VOC0712")
                    fileInfo.append(fl.split('.')[0]+'.jpg')
                flag = 1

                cnt += 1
                cor_dict = {}
                tmpv = []
                tmpv = contx.split(" ")

                if tmpv[0] == "person":
                    xmin = str(max(1,int(tmpv[1])))
                    ymin = str(max(1,int(tmpv[2])))
                    xmax = str(min(int(tmpv[1]) + int(tmpv[3]),640))
                    ymax = str(min(int(tmpv[2]) + int(tmpv[4]), 480))
                    cor_dict = {}
                    cor_dict["xmin"] = xmin
                    cor_dict["ymin"] = ymin
                    cor_dict["xmax"] = xmax
                    cor_dict["ymax"] = ymax
                    obj.append(cor_dict)
                # wf.write(str(tmpv))
                # wf.write('\n')
            else:
                if flag == 0:
                    fileInfo.append("VOC0712")
                    fileInfo.append(fl.split('.')[0]+'.jpg')
                flag = 1
        #train
        if flag == 1:
            #generate_xml("/home/user/PycharmProjects/caltech_new_anno/annos/", fileInfo, obj)
            #print fl
            #wf.write(fl.split('.')[0])
            # wf.write('\n')
            pass
        #test
        generate_xml("/home/user/PycharmProjects/caltech_new_anno/5_4annoss/", fileInfo, obj)
        wf.write(fl.split('.')[0])
        wf.write('\n')
    print cnt
    #wf.write(cnt)
    pass
    wf.close()

"""
这里替换不同的数据集，从而生成不同的ｘｍｌ文件
２０１８年０５月０４日２２：２１：４３　　对坐标进行了限制，即最大值和最小值
"""
#folder_struct(1, "/home/user/Downloads/caltech_data_set/datasets/caltechx10/train/annotations")
folder_struct(1, "/home/user/Downloads/caltech_data_set/datasets/caltechx1/train/annotations")
