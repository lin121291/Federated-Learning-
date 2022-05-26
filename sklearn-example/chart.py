import matplotlib.pyplot as plt

#還有server和no_FL沒加進來

f = open("output.txt","r")
d=f.readlines()

number=''
paint=[]

#處裡字串
for i in d:
    for j in i[3:]:
        if j==' ':
            paint.append(float(number))
            number=''
        else:
            number=number+j

    #畫圖
    print(paint)
    if i[0]=='c':
        if i[1]=='1':
            plt.plot(paint,color='r')
        else:
            plt.plot(paint,color='b')
    else:
        plt.plot(paint,color='k')
    
    paint=[]


f.close()
plt.show()
