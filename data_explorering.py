'''
Explorer the training file and generate traning/dev set 
'''

import random

import pandas as pd

df = pd.read_csv('Data/training.csv')

# To shuffle 'before' and 'after', generate 0 or 1 randomly
df['Label'] = df['BEFORE_BODY'].apply(lambda x: random.randint(0,1))

# Depending on label, generate X and Y
Text, Text_BODY = [], []
X, Y = [], []
X_BODY, Y_BODY = [], []
for i in range(len(df)):
    if df['Label'][i] == 0:
        X.append(str(df['AFTER_HEADLINE'][i])+'|'+df['AFTER_BODY'][i])
        Y.append(str(df['BEFORE_HEADLINE'][i])+'|'+df['BEFORE_BODY'][i])
        X_BODY.append(df['AFTER_BODY'][i])
        Y_BODY.append(df['BEFORE_BODY'][i])        
        Text.append('[CLS]'+str(df['AFTER_HEADLINE'][i])+'|'+df['AFTER_BODY'][i]+'[SEP]'+str(df['BEFORE_HEADLINE'][i])+'|'+df['BEFORE_BODY'][i])
        Text_BODY.append('[CLS]'+'|'+df['AFTER_BODY'][i]+'[SEP]'+'|'+df['BEFORE_BODY'][i])
    else:
        X.append(str(df['BEFORE_HEADLINE'][i])+'|'+df['BEFORE_BODY'][i])
        Y.append(str(df['AFTER_HEADLINE'][i])+'|'+df['AFTER_BODY'][i])
        X_BODY.append(df['BEFORE_BODY'][i])        
        Y_BODY.append(df['AFTER_BODY'][i])
        Text.append('[CLS]'+str(df['BEFORE_HEADLINE'][i])+'|'+df['BEFORE_BODY'][i]+'[SEP]'+str(df['AFTER_HEADLINE'][i])+'|'+df['AFTER_BODY'][i])
        Text_BODY.append('[CLS]'+df['BEFORE_BODY'][i]+'|'+'[SEP]'+'|'+df['AFTER_BODY'][i])
        
df['Text'] = Text
df['Text_BODY'] = Text_BODY
df['X'] = X
df['Y'] = Y
df['X_BODY'] = X_BODY
df['Y_BODY'] = Y_BODY


df_fin = df[['X','Y','X_BODY','Y_BODY','Label']]


# divide the dataset into train and dev sets
df_fin['prob'] = [random.random() for _ in range(len(df_fin))]

def split_df(prob):
    if prob < 0.8: return 'train'
    else: return 'dev'
    
df_fin['class'] = df_fin['prob'].apply(split_df)nn

df_fin_train = df_fin[df_fin['class']=='train'][['X','Y','X_BODY','Y_BODY','Label']]
df_fin_valid = df_fin[df_fin['class']=='dev'][['X','Y','X_BODY','Y_BODY','Label']]

# save as csv files without index
df_fin_train.to_csv('Data/train.csv', index=False)
df_fin_valid.to_csv('Data/dev.csv', index=False)