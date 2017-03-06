count=0
with open("actitracker_raw.txt") as f:
    for line in f:
        lineContent=line.split(",")
        # print(line)
        # print(len(lineContent))
        # lineContent[5]=lineContent[5].split(";")[1]
        # print(lineContent)
        # if lineContent[1]!="Walking" and lineContent[1]!="Jogging" and lineContent[1]!="Stairs" and lineContent[1]!="Sitting" and lineContent[1]!="Standing":
        #     print(lineContent)
        count += 1
        # if(len(lineContent)==6) and (count<10000 or (count>230039 and count<232640)):
        if(len(lineContent)==6) and (count<100000 or (count>200000 and count<300000)):
            f2=open("actitracker_100000.txt", "a")
            lineContent[5] = lineContent[5].split(";")[0]
            str=lineContent[0]+","+lineContent[1]+","+lineContent[2]+","+lineContent[3]+","+lineContent[4]+","+lineContent[5]+"\n"
            print(str)
            print(count)
            f2.write(str)