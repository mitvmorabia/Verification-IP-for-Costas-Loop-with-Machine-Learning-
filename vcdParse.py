
# coding: utf-8

# In[2]:


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

print(len(grps))
a = grps[60000:120000]
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
                            
df.to_csv("C:/Masters/297&298B/test_vcd12a.csv")            
print("done 10b")        
        
##for group in grps:
  #  if "1:" in gro
   
   
        

