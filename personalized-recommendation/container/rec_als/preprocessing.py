import pandas as pd
import numpy as np
import sys
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import json
import itertools

df=pd.read_csv('train_new_2.csv')
uuid=df['uuid'].to_list()
subcat_name=df['subcategory'].to_list()
attr=df['adattributes'].to_list()
price = []
for i in range(0, len(attr)):
    x = str(attr[i]).find("attributename=price")
    t = 0
    y = attr[i]
    if (x != -1):
        temp = y[x:]
        t = temp.find("}")
        res = y[x + 36:x + t]
        price.append(res)
    else:
        price.append(0)
brand_name=[]
for i in range(0,len(attr)):
    x=str(attr[i]).find("attributename=brand_name")
    t=0
    y=attr[i]
    if(x!=-1):
        temp=y[x:]
        t=temp.find("}")
        res=y[x+41:x+t]
        brand_name.append(res)
    else:
        brand_name.append("")
product_type=[]
for i in range(0,len(attr)):
    x=str(attr[i]).find("attributename=product_type")
    t=0
    y=attr[i]
    if(x!=-1):
        temp=y[x:]
        t=temp.find("}")
        res=y[x+43:x+t]
        product_type.append(res)
    else:
        product_type.append("")


appliance_type=[]
for i in range(0,len(attr)):
    x=str(attr[i]).find("attributename=appliance_type")
    t=0
    y=attr[i]
    if(x!=-1):
        temp=y[x:]
        t=temp.find("}")
        res=y[x+45:x+t]
        appliance_type.append(res)
    else:
        appliance_type.append("")
cat=df['category'].to_list()
product_type_final = []
for i in range(0, len(attr)):

    if product_type[i] == "":
        if appliance_type[i] == '':
            product_type_final.append(subcat_name[i])
        else:
            product_type_final.append(appliance_type[i])
    else:
        product_type_final.append(product_type[i])

df_new=pd.DataFrame({"uuid":uuid,"cat_name":cat,"subcat_name":subcat_name,"brand_name":brand_name,"price":price,"product_type":product_type,"appliance_type":appliance_type})
df_ea=df_new[df_new['cat_name']=='Electronics & Appliances'].reset_index()
brand_key_list=df_ea['brand_name'].value_counts().keys().tolist()
brand_value_list=df_ea['brand_name'].value_counts().tolist()
brand_dict = dict(zip(brand_key_list, brand_value_list))

out_dict_brand = dict(itertools.islice(brand_dict.items(), 20))

df_m=df_new[df_new['cat_name']=='Mobiles & Tablets'].reset_index()
brand_key_list_m=df_m['brand_name'].value_counts().keys().tolist()
brand_value_list_m=df_m['brand_name'].value_counts().tolist()
brand_dict_m = dict(zip(brand_key_list_m, brand_value_list_m))
out_dict_brand_m = dict(itertools.islice(brand_dict_m.items(), 20))


def get_price_band(price):
    price = int(price)
    if (price < 2000):
        return '<2k'
    elif (price >= 2000 and price < 5000):
        return '2k-5k'
    elif (price >= 5000 and price < 8000):
        return '5k-8k'
    elif (price >= 8000 and price < 10000):
        return '8k-10k'
    elif (price >= 10000 and price < 15000):
        return '10k-15k'
    else:
        return '>15k'


def get_price_min_max(price):
    op_less_than = price.find('<')
    op_greater_than = price.find('>')
    # print()
    op_sperator = price.find('-')
    price_min = ''
    price_max = ''
    if price == '0':
        price_min = ''
        price_max = ''
        return price_min, price_max

    elif (op_less_than >= 0):
        price_min = '0'
        price_max = '2000'
        return price_min, price_max
    elif op_greater_than >= 0:
        # print("Greater")
        price_min = '15000'
        price_max = ''
        return price_min, price_max
    elif op_sperator > 0:
        price_min = str(int(price[0:op_sperator - 1]) * 1000)
        price_max = str(int(price[op_sperator + 1:len(price) - 1]) * 1000)

    return price_min, price_max


feature = []
for i in range(0, len(attr)):
    # brand=''
    attr = {}
    if cat[i] == 'Mobiles & Tablets':
        if (brand_name[i] == '' or brand_name[i] == 'Other'):
            brand_name[i] = "Others"

        if brand_name[i] in out_dict_brand_m.keys():
            brand = brand_name[i]
        else:
            brand_name[i] = "Others"

        price_1 = get_price_band(price[i])
        price_min, price_max = get_price_min_max(price_1)

        attr["brand_name"] = brand_name[i]
        attr["cat_name"] = cat[i]
        attr["subcat_name"] = subcat_name[i]
        price_attr = {"price_min": price_min, "price_max": price_max}
        # price_attr=json.dumps(price_attr)
        attr["price"] = price_attr
        result = json.dumps(attr)
        feature.append(result)


    elif cat[i] == 'Electronics & Appliances':
        if (brand_name[i] == '' or brand_name[i] == 'Other'):
            brand_name[i] = "Others"

        if brand_name[i] in out_dict_brand.keys():
            brand = brand_name[i]
        else:
            brand_name[i] = "Others"
        attr["brand_name"] = brand_name[i]
        attr["cat_name"] = cat[i]
        attr["subcat_name"] = subcat_name[i]
        attr["product_type"] = product_type_final[i]
        if (attr["product_type"] == ''):
            if attr["appliance_type"] != "":
                attr["appliance_type"] = appliance_type[i]
        # price_attr={"price_min":price_min,"price_max":price_max}
        # attr["price"]=price_attr
        result = json.dumps(attr)
        feature.append(result)
        # feature.append((brand_name[i])+" "+str(product_type_final[i]))

    else:
        feature.append('')

df_final=pd.DataFrame({"uuid":uuid,"feature":feature,"cat_name":cat})
df_final1=df_final[df_final['cat_name'].notna()]
df_2=df_final1.groupby(['uuid','feature']).count()
df_2.to_csv('new_train_data.csv',header=None)
df_train=pd.read_csv('new_train_data.csv',header=None)
df_train.columns=[['uuid','feature','score']]
df_train.to_csv('s3://bucket_name/training_data.csv')




