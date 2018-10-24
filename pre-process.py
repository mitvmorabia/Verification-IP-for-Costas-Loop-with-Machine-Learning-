
# coding: utf-8

# In[1]:


import pandas as pd

import itertools
import re

f = open("C:/Masters/297&298B/ML/costas.vcd")
print(f)
lines = f.readlines()
# count = 0
# for words in lines:
# for i in range(2550,len(lines)):
#     if '$dumpvars' in lines[i]:
#         extract(i,lines)
#     else:
#         print("flop")
start_in = [i for i in range(len(lines)) if '$dumpvars' in lines[i]][0]
reqd_lines = lines[start_in+1:]

def bobo(x):    
    if x.startswith('#'):
        bobo.count+=1
    return bobo.count
bobo.count=0

grps = []

for key,grp in itertools.groupby(reqd_lines,bobo):
#     print(key,list(grp))

    grps.append(list(grp))

b = len(grps) - 1
a = grps[60000:b]
df = pd.DataFrame(columns=['time','bitStrobe','pushADC', 'sincos'])
print(df)
time = ''
value = ''
#mValue = ''
#scwkVal = ''
sincosVal = ''
pushVal =''
#sincos_dVal = ''
for elements in a:
    true = 0
    for command in elements:
        a = re.search(r"#\d",command)
        b = re.search(r"\d+:\s",command)
        #c = re.search(r";\s",command)
        sincos = re.search(r"m\s",command)
        pushADC = re.search(r"\dd\s", command)
       
        #scwk = re.search(r"l\s",command)
        #sincos_d = re.search(r"n\s",command)
       
        #if (a or b or c or sincos or scwk or sincos_d):
        if (a or b or sincos or pushADC):
            #print(command)
            if (a):
                time = command.split("#")[1].strip('\n')
                #print(type(time))
            if (b):
                value = command.split(":")[0]
                #print(type(value))
            #if (c):
            #    mValue = command.split(";")[0]
            if (sincos):
                sincosVal = command.split("m")[0]
            if(pushADC):
                pushVal = command.split("d")[0]
            #if (scwk):
            #    scwkVal = command.split("l")[0]
            #if (sincos_d):
            #    sincos_dVal = command.split("n")[0]
             
                #print(type(mValue))
            #print(time, value, mValue)
            new_row = [time, str(value),str(pushVal), str(sincosVal) ]
            df = df.append(pd.Series(new_row, index=df.columns), ignore_index=True)
                            
df.to_csv("C:/Masters/297&298B/test_vcd_complete.csv")            
print("Complete file ready")        


# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv("C:/Masters/297&298B/test_vcd_complete.csv")
df["sincos"] = df["sincos"].str.replace("b","")

#make index unique
new = df.index[df['bitStrobe'] == True].tolist()
new1 = []
new1.append(new[0])
for i in range(len(new)-1):
    if (new[i+1]-new[i])>2:
        new1.append(new[i+1])
print(new1)


new_time = df.index[df['bitStrobe'] == True].tolist()
first_bit_index = new_time[0]

time_start = df['time'][first_bit_index] 
print("this is time start {}".format(time_start))
print("\n \n \n")
time_jump = 250000
push_jump = 20000
new_df_aslist = []
df_group_length = []
attr_val_row = []
attr_val_row_length = []

for elements in new1:
    attr_val = []
    attr_val_length = []

    print("this is main element {} \n".format(df['time'][elements]))
    minus = df['time'][elements] - time_jump
    print("this is minus {} \n".format(minus))
    plus = df['time'][elements] + time_jump
    print("this is plus {} \n".format(plus))
    
    
    df_start_index = df.index[df['time'] == minus].tolist()
    df_end_index = df.index[df['time'] == plus].tolist()
    print(df_start_index, df_end_index)
    if(len(df_end_index)) >0:
        pass
    else:
        break
    
    df_group = df[df_start_index[0] : df_end_index[0]]
    print(len(df_group))
    ## start to group attributes ( eye )
    start_time = df_group['time'].iloc[0]
    end_time = df_group['time'].iloc[-1]
    print("this is start time {0} and this is end time {1}".format(start_time, end_time))
    push_val = start_time
    
    for i in range(25):
        push_val = push_val + push_jump
        print('this is push val {}'.format(push_val))
        if(push_val <= end_time):
            df_group_index = df_group.index[df_group['time'] == push_val].tolist()
            df_group_index = int(df_group_index[0])
            #print(df_group_index)
            sincos_val = df_group['sincos'][df_group_index]
            #print(sincos_val)
            attr_val.append(sincos_val)
            attr_val_length.append(attr_val_length)
        else:
            print("help")
    attr_val_row.append(attr_val)
    attr_val_row_length.append(len(attr_val_row[0]))
        
    #print(start_time, end_time)
print("this is my final {} \n".format(attr_val_row))    
print("this is my final list length {}".format(attr_val_row_length))

    



# In[ ]:


def bin2int(b):
    c = int(b,2)
    if c> 2**(31):
        d = c - 2**32
        return d
    else:
        return c



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l)-1, n):
        mit1 = l[i:i + n]
        df_ml_parts.append(mit1)
    return(df_ml_parts)    
df_ml =[]
df_ml_parts = []
count = 0
for i in attr_val_row:
    mit = chunks(i,6)
    df_ml.append(mit)

    # 6 works

#print(df_ml)
mydf= pd.DataFrame(columns =['1st', '2nd','3rd','4th','5th','6th'])
for sublist in df_ml:
    for item in sublist:
        mydf.loc[len(mydf)] = item
print(mydf)

mydf['1st'] = mydf['1st'].apply(lambda x: bin2int(x))
mydf['2nd'] = mydf['2nd'].apply(lambda x: bin2int(x))
mydf['3rd'] = mydf['3rd'].apply(lambda x: bin2int(x))
mydf['4th'] = mydf['4th'].apply(lambda x: bin2int(x))
mydf['5th'] = mydf['5th'].apply(lambda x: bin2int(x))
mydf['6th'] = mydf['6th'].apply(lambda x: bin2int(x))
#mydf['5th'] = mydf['5th'].apply(lambda x: bin2int(x))
print(mydf)
mydf.to_csv("C:/Masters/297&298B/training_data_complete.csv")
''' 
    for j in range(5):
        first = 4*j
        second = (4*j) + 1
        string1 = i[first:second]
    #    print(string1)
        df_ml.append(string1)

print(df_ml) ''' 


# In[ ]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
#import step, show
import numpy as np
from scipy import interpolate


print(len(mydf))
a = np.array_split(mydf, len(mydf)/8)
print(len(a))

#only multiples of 0,6, 12,18, 24,  give good eye structures

df1 = a[6]
print(df1)
ax = df1.plot.line()
ax.set_title('Before interpolation')
ax.set_xlabel("time")
ax.set_ylabel("value")
 
df1 = df1[df1["1st"] < 400]
df1 = df1[df1["1st"] > -400]

df1 = df1[df1["2nd"] < 400]
df1 = df1[df1["2nd"] > -400]

df1 = df1[df1["3rd"] < 400]
df1 = df1[df1["3rd"] > -400]

df1 = df1[df1["4th"] < 400]
df1 = df1[df1["4th"] > -400]

df1 = df1[df1["5th"] < 400]
df1 = df1[df1["5th"] > -400]

df1 = df1[df1["6th"] < 400]
df1 = df1[df1["6th"] > -400]

df2 = df1.transpose()

ax2 = df2.plot.line()
ax2.set_title('After interpolation')
ax2.set_xlabel("year")
ax2.set_ylabel("weight")

#f1 = interp1d(df1.index, df1['1st'],kind='cubic')
#f2 = interp1d(df1.index, df1['2nd'],kind='cubic')
#f3 = interp1d(df1.index, df1['3rd'],kind='cubic')
plt.show()




