'''
--------------------------------------------------
File: demographic_models.py
--------------------------------------------------
Author: Deloitte Australia 2021

Description: Defines a basic matrix factorisation model for user to product recommendations


--------------------------------------------------
Edit History:

#  | NAME				|  DATE       	| DESC
0  | Grant Holtes       |  26/07/21		| Initial Creation
1  | Lilly Taraborelli  |  11/08/21		| Implimentation
2  | Lilly Taraborelli  |  30/08/21		| Update to optimise code (used groupby, removed functions and changed def find_most_similar_customer) - as per Grant's feedback
3  | Lilly Taraborelli  |  01/09/21		| Updated customer_vector function
--------------------------------------------------
'''
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import json
from scipy.spatial import distance
import random 
import functools

orderDataPath = 'recommendation-engine\src\models\data\orders_dataset.csv' 
productDataPath = 'recommendation-engine\src\models\data\products_dataset_with_imgs.csv'
unknown_customer_JSON = 'recommendation-engine\src\models\data\CC.json'

@functools.lru_cache(maxsize=1) #Caches results
def update_customer_vectors(orderDataPath, productDataPath):
    #read csvs
    df_orders = pd.read_csv(orderDataPath) #path to enter is 'data\orders_dataset.csv' 
    df_products = pd.read_csv(productDataPath) #path to enter is 'data\products_dataset_with_imgs.csv'

    #convert csv to DF with selected columns
    sub_orders = df_orders[["customer_id", "product_id"]]
    sub_products = df_products[["product_id", "product_category_name_eng"]]

    #join orders and products to get product catergory
    order_product = pd.merge(sub_orders, sub_products)

    #drop product_id column
    order_product = order_product.drop(['product_id'], axis = 1)

    #create transition matrix using grouped by
    order_product_groupedby = order_product.groupby(['customer_id', 'product_category_name_eng']).size().unstack('product_category_name_eng', fill_value=0)
    order_product_groupedby.reset_index(inplace=True)
    order_product_numpy = order_product_groupedby.to_numpy()

    return order_product_numpy


def make_vector(productDataPath, unknown_customer_JSON):
    #read csv - copied this from above because I can't call on 'df_products' as it is stored in a function.
    df_products = pd.read_csv(productDataPath) #path to enter is 'data\products_dataset_with_imgs.csv'

    #open and load JSON 
    with open(unknown_customer_JSON) as CCjson: # path is 'data\'CC.json'
        UnknownCC_JSON = json.load(CCjson)

    #make a list of product that Unknown CC clicked on
    UnknownCC_viewed_products = []
    for event in UnknownCC_JSON['events']:
        UnknownCC_viewed_products.append(event['props']['product']['label'])

    #transform list to DF, and join with product data to concatenate category
    UnknownCC_product_dict= {'title':UnknownCC_viewed_products}
    UnknownCC_DF = pd.DataFrame(data=UnknownCC_product_dict)
    master_product_data = df_products[["product_category_name_eng", 'title']]
    UnknownCC_products_DF = (pd.merge(UnknownCC_DF, master_product_data)).drop('title', axis=1)

    #create a list of unique category names
    unique_categories = []
    for x in df_products['product_category_name_eng'].unique():
        unique_categories.append(x)
    unique_categories.sort()

    #create a structured array for the category names
    DTd = []
    for x in unique_categories:
        DT = (x, 'i8')
        DTd.append(DT)
    UnknownCC_vector = np.zeros((1), dtype=DTd)

    #populate vector
    for x in UnknownCC_products_DF['product_category_name_eng']:
        for category in unique_categories:
            if x == category:
                UnknownCC_vector[category] = UnknownCC_vector[category] + 1
    
    #change numpy void to array (required for later cosine distance comparison)
    UnknownCC_vector_clean = []
    for x in UnknownCC_vector[0]:
        UnknownCC_vector_clean.append(x)
    UnknownCC_vector_clean=np.array(UnknownCC_vector_clean)
        
    return UnknownCC_vector_clean


known_customer_vector = update_customer_vectors(orderDataPath, productDataPath)
unknown_customer_vector = make_vector(productDataPath, unknown_customer_JSON)

def find_most_similar_customer(known_customer_vector, unknown_customer_vector):    
    most_similar_known_cust = 'UNK'
    max_similarity = 0
    
    #find highest similarity 
    for i in known_customer_vector:
        #calculate simliarity 
        cs_similarity = 1 - (distance.cosine(i[1:], unknown_customer_vector))
        #stores max similarity 
        if cs_similarity > max_similarity:
            max_similarity = cs_similarity
            most_similar_known_cust = i[0]
    #find corresponding vector for most similar customer
    dict = {}
    for x in update_customer_vectors(orderDataPath, productDataPath):
        dict[x[0]] = x[1:]
    #return most similar customer and their vector
    return most_similar_known_cust, dict[most_similar_known_cust]

print(find_most_similar_customer(known_customer_vector, unknown_customer_vector))

#im not sure what to change here
# if __name__ == "__main__":
#     customerDataPath = "PATH/TO/DATA"
#     update_customer_vectors(customerDataPath)