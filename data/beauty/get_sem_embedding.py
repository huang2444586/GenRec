import os
import pickle
import random
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm
from openai import OpenAI

os.environ["DASHSCOPE_API_KEY"] = 'sk-fdc1c9908fde4e8496d6e2a501137950'
PROMPT_TEMPLATE = "The beauty item has following attributes: \n name is <TITLE>; brand is <STORE>; price is <PRICE>. \n"
FEAT_TEMPLATE = "The item has following features: <FEATURES>. \n"
DESC_TEMPLATE = "The item has following description: <DESCRIPTION>. \n"
SEED = 666
DATASET_NAME = 'beauty'
USED_DIR = os.path.join('.', 'data', DATASET_NAME, 'handle')

def get_attri(item_str, attri, item_info):
    """
    将物品属性嵌入提示模板
    """
    if attri not in item_info.keys():
        new_str = item_str.replace(f"<{attri.upper()}>", "unknown")
    else:
        new_str = item_str.replace(f"<{attri.upper()}>", str(item_info[attri]))

    return new_str

def get_feat(item_str, feat, item_info):
    """
    将物品特征嵌入提示模板
    """
    if feat not in item_info.keys():
        return ""
    
    assert isinstance(item_info[feat], list)
    feat_str = ""
    for meta_feat in item_info[feat]:
        feat_str = feat_str + meta_feat + ";"
    new_str = item_str.replace(f"<{feat.upper()}>", feat_str)

    # if len(new_str) > 2048: # avoid exceed the input length limitation
    #     return new_str[:2048]

    return new_str

def get_response(prompt):
    """
    从API获取文本嵌入
    """

    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=os.getenv("DASHSCOPE_API_KEY"),  
        # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=prompt
    )

    re_json = json.loads(completion.model_dump_json())
    return re_json["data"][0]["embedding"]


def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open(os.path.join(USED_DIR, data_path), "w") as w:
        for meta_data in data:
            w.write(meta_data)


def main():
    data = json.load(open(os.path.join(USED_DIR,'asin2metainfo.json'), "r"))

    # all_attris = []  # 读取item所有属性
    # for user, user_attris in data.items():
    #     for feat_name in user_attris.keys():
    #         if feat_name not in all_attris:
    #             all_attris.append(feat_name)

    item_data = {}
    for key, value in tqdm(data.items()):
        item_str = copy.deepcopy(PROMPT_TEMPLATE)
        item_str = get_attri(item_str, "title", value)
        item_str = get_attri(item_str, "store", value)
        item_str = get_attri(item_str, "price", value)

        feat_str = copy.deepcopy(FEAT_TEMPLATE)
        feat_str = get_feat(feat_str, "features", value)
        desc_str = copy.deepcopy(DESC_TEMPLATE)
        desc_str = get_attri(desc_str, "description", value)
        
        item_data[key] = item_str + feat_str + desc_str

    # random_items = dict(random.sample(list(item_data.items()), min(10, len(item_data))))
    # print("随机输出10个:")
    # for key, value in random_items.items():
    #     print(f"Key: {key}, Value: {value}...")
    """
    Key: B00NV6ZLJK, Value: The beauty item has following attributes: 
    name is Seven Stars Silver Plated Thin Chain Anklet Bracelet Foot Jewelry Barefoot; brand is Hittime; price is None. 
    The item has following features: Match with suitable apparel for different occasion.; Wonderful gift for you and your femal friends.; Catch this beautiful accessories for you.; Material: Alloy.; Size : 20+3cm (adjustable); . 
    The item has following description: ['Features:  Brand new and high quality. Stylish Hearts chain anklet, exclusive and fashionable for girls and ladies. Nice accessories to integrate jewelry case for girls and collectors. Match with suitable apparel for different occasion. Wonderful gift for you and your femal friends. Catch this beautiful accessories for you. Material: Alloy.Size : 20+3cm (adjustable)  Package Included:  1 x Anklet  NO Retail Box. Packed Safely in Bubble Bag.'].
    """

    json.dump(item_data, open(os.path.join(USED_DIR,'item_str.json'), "w"))
    # item_data = json.load(open("./handled/item_str.json", "r"))

    id_map = json.load(open(os.path.join(USED_DIR,'id_map.json'), "r"))["item2id"]
    json_data = []
    for key, value in item_data.items():
        json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

    save_data("item_str.jsonline", json_data)

    item_emb = {}

    if os.path.exists(os.path.join(USED_DIR,'item_emb.pkl')):    # check whether some item emb exist in cache
        item_emb = pickle.load(open(os.path.join(USED_DIR,'item_emb.pkl'), "rb"))

    count = 1
    while 1:    # avoid broken due to internet connection
        if len(item_emb) == len(item_data):
            break
        try:
            for key, value in tqdm(item_data.items()):
                if key not in item_emb.keys():
                    if len(value) > 4096:
                        value = value[:4096]
                    item_emb[key] = get_response(value)
                    count += 1
        except:
            pickle.dump(item_emb, open(os.path.join(USED_DIR,'item_emb.pkl'), "wb"))
            exit()
    
    print('item_emb LENGHT:', len(item_emb))
    print('item_data LENGHT:', len(item_data))

    id_map = json.load(open(os.path.join(USED_DIR,'id_map.json'), "r"))["id2item"]
    emb_list = []
    for id in range(1, len(item_emb)+1):
        meta_emb = item_emb[id_map[str(id)]]
        emb_list.append(meta_emb)

    emb_list = np.array(emb_list)
    pickle.dump(emb_list, open(os.path.join(USED_DIR,'item_emb_np.pkl'), "wb"))

if __name__ == "__main__":
    # emb = get_response("How to make a good coffee?")
    # print(emb)
    main()