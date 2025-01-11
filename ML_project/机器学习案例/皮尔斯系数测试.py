import pandas as pd

data = {
    'a':[1,2,3,4,5],
    'b':[6,7,8,3,2]
}
data2 = {
    'a':[1,2,3,4,5],
    'b':[1,4,9,16,25]
}

data3 = {
    'a':[1,2,3,4,5],
    'b':[False,False,False,True,True]
}

df = pd.DataFrame(data)
print(df)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
print('*'*32)
print(len(df))
print('*'*32)

coscore = df.corr()
coscore2 = df2.corr()
coscore3 = df3.corr()

print(coscore)
print(coscore2)
print(coscore3)