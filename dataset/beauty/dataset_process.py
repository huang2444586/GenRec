from collections import defaultdict
import gzip
import json
import os
import numpy as np
from tqdm import tqdm

PARENT = os.path.dirname
DATASET_NAME = 'beauty'
SEED = 666
LOAD_DIR = os.path.join('.', 'dataset', DATASET_NAME)
SAVE_DIR = os.path.join('.', 'data', DATASET_NAME, 'handle')
INPUT_FILES = [
    'All_Beauty.jsonl.gz',
    'meta_All_Beauty.jsonl.gz'
]

def parse(path):
    # read file.gz to json format
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        inter_list.append(json.loads(l.decode()))
    return inter_list


def Amazon(file_name, rating_score):
    """
    get review data like (user, item, time) from
    {'rating': 5.0, 
     'title': 'Such a lovely scent but not overpowering.', 
     'text': "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!", 
     'images': [], 
     'asin': 'B00YQ6X8EO', 
     'parent_asin': 'B00YQ6X8EO', 
     'user_id': 'AGKHLEW2SOWHNMFQIJGBECAF7INQ', 
     'timestamp': 1588687728923, 
     'helpful_vote': 0, 
     'verified_purchase': True}
    """
    datas = []
    data_flie = os.path.join(LOAD_DIR, file_name)
    for inter in parse(data_flie):
        if float(inter['rating']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['user_id']
        item = inter['asin']
        time = inter['timestamp']
        datas.append((user, item, int(time)))
    return datas

def Amazon_meta(file_name):
    """ get (id, item info) from
    {'main_category': 'All Beauty',
     'title': 'Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)',
     'average_rating': 4.8, 
     'rating_number': 10, 
     'features': [], 
     'description': [], 
     'price': None, 
     'images': [{'thumb': 'https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg', 'large': 'https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg', 'variant': 'MAIN', 'hi_res': None}, {'thumb': 'https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg', 'large': 'https://m.media-amazon.com/images/I/41w2yznfuZL.jpg', 'variant': 'PT01', 'hi_res': 'https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg'}], 
     'videos': [], 
     'store': 'Howard Products', 
     'categories': [], 
     'details': {'Package Dimensions': '7.1 x 5.5 x 3 inches; 2.38 Pounds', 'UPC': '617390882781'}, 
     'parent_asin': 'B01CUPMQZE', 
     'bought_together': None}
    """
    datas = {}
    meta_flie = os.path.join(LOAD_DIR, file_name)
    for info in tqdm(parse(meta_flie)):
        datas[info['parent_asin']] = info
    return datas

def fliter_meta(meta_infos, data_maps):
    datas = {}
    item_asins = list(data_maps['item2id'].keys())
    for asin, info in tqdm(meta_infos.items()):
        if asin not in item_asins:
            continue
        datas[asin] = info
    return datas

def get_attribute_Amazon(meta_infos, datamaps, attribute_core):
    """
    get attribute info from meta_infos
    """
    attribute2id = {}  # (attribute_name, attribute_id)
    id2attribute = {}  # (attribute_id, attribute_name)
    attributeid2num = defaultdict(int)  # (attribute_id, occurrence_num)
    attribute_id = 1  # init the attribute_id
    items2attributes = {}  # (item_id, [attribute_id])
    attribute_lens = []  # (item_id, attribute_list_len)

    for iid, attributes in meta_infos.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # update datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def filter_common(user_items, user_t, item_t):
    """
    filter common items and users
    """
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, item, _ in user_items:
        user_count[user] += 1
        item_count[item] += 1

    User = {}
    for user, item, timestamp in user_items:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User.keys():
            User[user] = []
        User[user].append((item, timestamp))

    new_User = {}
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[1])
        new_hist = [i for i, t in User[userid]]
        new_User[userid] = new_hist

    return new_User

def id_map(user_items): 
    """
    id map
    """
    user2id = {} # user asin 2 uid
    item2id = {} # item asin 2 iid
    id2user = {} # uid 2 user asin
    id2item = {} # iid 2 item asin
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps

def get_counts(user_items):
    """
    get user and item counts
    """
    user_count = {}
    item_count = {}

    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count.keys():
                item_count[item] = 1
            else:
                item_count[item] += 1

    return user_count, item_count


def filter_minmum(user_items, min_len=3):
    """
    filter the user-item interaction which less than min_len
    """
    new_user_items = {}
    for user, items in user_items.items():
        if len(items) >= min_len:
            new_user_items[user] = items

    return new_user_items

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def unify_data(interactions, meta_infos):
    """
    统一交互数据和元数据，删除多余部分
    
    Args:
        interactions: 交互数据列表 [(user, item, timestamp), ...]
        meta_infos: 元数据字典 {item_id: item_info, ...}
        
    Returns:
        unified_interactions: 统一后的交互数据
        unified_meta_infos: 统一后的元数据
    """
    # 获取交互数据中的所有物品ID
    interaction_items = set()
    for _, item, _ in interactions:
        interaction_items.add(item)
    
    # 获取元数据中的所有物品ID
    meta_items = set(meta_infos.keys())
    
    # 找到交集 - 同时存在于交互数据和元数据中的物品
    common_items = interaction_items.intersection(meta_items)
    
    print(f"交互数据中的物品数: {len(interaction_items)}")
    print(f"元数据中的物品数: {len(meta_items)}")
    print(f"同时存在于两者的物品数: {len(common_items)}")
    print(f"只在交互数据中存在的物品数: {len(interaction_items - meta_items)}")
    print(f"只在元数据中存在的物品数: {len(meta_items - interaction_items)}")
    
    # 过滤交互数据，只保留在common_items中的物品
    unified_interactions = []
    for user, item, timestamp in interactions:
        if item in common_items:
            unified_interactions.append((user, item, timestamp))
    
    # 过滤元数据，只保留在common_items中的物品
    unified_meta_infos = {}
    for item_id, item_info in meta_infos.items():
        if item_id in common_items:
            unified_meta_infos[item_id] = item_info
            
    return unified_interactions, unified_meta_infos


def main(data_name, user_core=3, item_core=3):
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    attribute_core = 0
    np.random.seed(SEED)

    review_file_name = INPUT_FILES[0]
    meta_file_name = INPUT_FILES[1]
    datas = Amazon(review_file_name, rating_score=rating_score)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    meta_infos = Amazon_meta(meta_file_name)
    print(f'{data_name} Meta data has been processed!')

    datas, meta_infos = unify_data(datas, meta_infos)

    # user_items = get_interaction(datas)
    user_items = filter_common(datas, user_t=user_core, item_t=item_core)
    # user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items = filter_minmum(user_items, min_len=3)
    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    # user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count, item_count = get_counts(user_items)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    meta_infos = fliter_meta(meta_infos, data_maps)
    attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    handled_path = os.path.join(SAVE_DIR)
    if not os.path.exists(handled_path):
        os.makedirs(handled_path)

    data_file = os.path.join(handled_path, 'inter_seq.txt')
    item2attributes_file = os.path.join(handled_path, 'asin2metainfo.json')
    id_file = os.path.join(handled_path, 'id_map.json')

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(meta_infos)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    with open(id_file, "w") as f:
        json.dump(data_maps, f)


if __name__ == '__main__':
    main(DATASET_NAME, 3, 2)


