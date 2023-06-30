from __future__ import print_function
import warnings
from models_code import CodeMF
from configs import *
from sklearn.metrics import *
import logging
# 给语料打标签
import os
import tensorflow as tf
import numpy as np
import random
import pickle
import argparse
random.seed(42)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


# from models import CodeMF
# from models_text import CodeMF

warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

set_session = tf.compat.v1.keras.backend.set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # half of the memory
set_session(tf.compat.v1.Session(config=config))

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class StandoneCode:
    # dict.get(）：返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self._buckets = conf.get('buckets', [(
            2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max(
            [j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max(
            [j for _, _, _, j in self._buckets]))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

    ##### Data Set #####

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' +
                        self.model_params['model_name'] + '/')
        model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch), overwrite=True)

    def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        print(self.path)
        print("{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, d12, d3, d4, d5, r, epoch))
        assert os.path.exists(
            "{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path, d12, d3, d4, d5, r, epoch)), "Weights at epoch {:d} not found".format(epoch)
        model.load_weights("{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, d12, d3, d4, d5, r, epoch))

    def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
        if (len((prepoch))) >= 2:
            lenth = len(prepoch)
            epoch = prepoch[lenth-2]
            if os.path.exists("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch)):
                os.remove("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                    self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch))

    def process_instance(self, instance, target, maxlen):
        w = self.pad(instance, maxlen)
        target.append(w)

    def process_matrix(self, inputs, trans1_length, maxlen):
        inputs_trans1 = np.split(inputs, trans1_length, axis=1)
        processed_inputs = []
        for item in inputs_trans1:
            item_trans2 = np.squeeze(item, axis=1).tolist()
            processed_inputs.append(item_trans2)
        return processed_inputs


def get_data(self, path):
    data = self.load_pickle(path)

    text_S1 = []
    text_S2 = []
    code = []
    queries = []
    labels = []
    ids = []

    text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
    text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                      text_block_length, 100)

    text_S1 = text_blocks[0]
    text_S2 = text_blocks[1]

    code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                      text_block_length - 1, 350)
    code = code_blocks[0]

    queries = [samples_term[3] for samples_term in data]
    labels = [samples_term[5] for samples_term in data]
    ids = [samples_term[0] for samples_term in data]

    return text_S1, text_S2, code, queries, labels, ids


def eval(self, model, path):
    text_S1, text_S2, code, queries, labels, ids = self.get_data(path)

    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred = np.argmax(labelpred, axis=1)

    loss = log_loss(labels, labelpred)
    acc = accuracy_score(labels, labelpred)
    f1 = f1_score(labels, labelpred)
    recall = recall_score(labels, labelpred)
    precision = precision_score(labels, labelpred)
    print("测试性能: precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f" % (
        precision, recall, f1, acc))
    return acc, f1, recall, precision, loss


def u2l_codemf(self, model, path, save_path):
    total_label = []
    text_S1, text_S2, code, queries, labels, ids = self.get_data(path)
    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred = np.argmax(labelpred, axis=1)

    total_label.append(ids)
    total_label.append(labelpred.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("codemf标签已打完")


def u2l_textsa(self, model, path, save_path):
    with open(save_path, 'r') as f:
        pre = eval(f.read())
    my_pre1 = pre[1]  # codemf_label
    total_label = []
    text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)
    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred1 = np.argmax(labelpred, axis=1)

    total_label.append(ids1)
    total_label.append(my_pre1)
    total_label.append(labelpred1.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("textsa标签已打完")


def u2l_codesa(self, model, path, save_path):
    with open(save_path, 'r') as f:
        pre = eval(f.read())
    my_pre1 = pre[1]  # codemf_label
    my_pre2 = pre[2]  # textsa_label

    total_label = []
    text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)
    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred1 = np.argmax(labelpred, axis=1)

    total_label.append(ids1)
    total_label.append(my_pre1)
    total_label.append(my_pre2)
    total_label.append(labelpred1.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("codesa标签已打完")


def final_analyze(path, hnn_path, save_path):
    with open(path, 'r') as f:
        pre = eval(f.read())
    ids = pre[0]
    codemf_label = pre[1]
    textsa_label = pre[2]
    codesa_label = pre[3]
    hnn_label_1 = []
    with open(hnn_path, 'r') as f:
        hnn = eval(f.read())
    hnn_ids = hnn[0]
    hnn_label = hnn[1]
    for i in range(len(hnn_ids)):
        if hnn_label[i] == 1:
            hnn_label_1.append(hnn_ids[i])

    total_final = []
    count = 0
    for i in range(len(ids)):
        if codesa_label[i] == 1 and textsa_label[i] == 1 and codemf_label[i] == 1:
            if ids[i] in hnn_ids:
                continue
            else:
                total_final.append(ids[i])
                count += 1

    total_final = total_final + hnn_label_1

    with open(save_path, "w") as f:
        f.write(str(len(total_final)) + '\n')
        for id_ in total_final:
            f.write(str(id_) + '\n')

    print("标签分析完成")


def final_analay_large(path, hnn_path, single_path, save_path):
    with open(path, 'r') as f:
        pre = eval(f.read())
    ids = pre[0]
    codemf_label = pre[1]
    textsa_label = pre[2]
    codesa_label = pre[3]
    hnn_label_1 = []
    with open(hnn_path, 'r') as f:
        hnn = eval(f.read())
    hnn_ids = hnn[0]
    hnn_label = hnn[1]
    for i in range(len(hnn_ids)):
        if hnn_label[i] == 1:
            hnn_label_1.append(hnn_ids[i])

    total_final = []
    count = 0
    for i in range(len(ids)):
        if codesa_label[i] == 1 and textsa_label[i] == 1 and codemf_label[i] == 1:
            if ids[i] in hnn_ids:
                continue
            else:
                total_final.append(ids[i])
                count += 1

    with open(single_path, 'r') as f:
        single = eval(f.read())
    single_ids = []
    for i in range(len(single)):
        single_ids.append(single[i][0])

    total_final = total_final + hnn_label_1 + single_ids

    with open(save_path, "w") as f:
        f.write(str(len(total_final)) + '\n')
        for id_ in total_final:
            f.write(str(id_) + '\n')

    print("标签分析完成")


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Model")
    parser.add_argument(
        "--train", choices=["python", "sql"], default="sql", help="train dataset set")
    parser.add_argument("--mode", choices=["train", "eval"], default='eval',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set ")
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = get_config_u2l(args.train)
    train_path = conf['data_params']['train_path']
    dev_path = conf['data_params']['valid_path']
    test_path = conf['data_params']['test_path']
    embding = conf['data_params']['code_pretrain_emb_path']

    ##### Define model ######
    logger.info('Build Model')

    model = eval(conf['model_params']['model_name'])(conf)
    StandoneCode = StandoneCode(conf)

    # ====================================sql打标签====================================
    staqc_sql_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    large_single_path = '../data_processing/hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    hnn_lable_sql_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_sql.txt'
    staqc_sql_final_label = '../data_processing/hnn_process/ulabel_data/staqc/sql_final_label.txt'
    save_path_final_label_staqc_sql = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_sql_label.txt'

    large_sql_final_label = '../data_processing/hnn_process/ulabel_data/staqc/large_sql_final_label.txt'
    save_path_final_label_large_sql_mul = '../data_processing/hnn_process/ulabel_data/staqc/combine_codedb_sql_final_label_mul.txt'

    # ====================================python打标签====================================
    staqc_python_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    large_single_python_path = '../data_processing/hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    hnn_lable_python_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_python.txt'
    staqc_python_final_label = '../data_processing/hnn_process/ulabel_data/staqc/python_final_label.txt'
    save_path_final_label_staqc_python = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_python_label.txt'

    large_python_final_label = '../data_processing/hnn_process/ulabel_data/staqc/large_python_final_label.txt'
    save_path_final_label_large_python_mul = '../data_processing/hnn_process/ulabel_data/staqc/combine_codedb_python_final_label_mul.txt'

    drop1 = drop2 = drop3 = drop4 = drop5 = np.round(0.25, 2)
    model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                        Regularizer=round(0.0004, 4), num=8, seed=42)

    model.build()
    if args.mode == 'eval':
        if args.train == "sql":
            # 第一次打标签
            print("第一次打标签开始：")
            u2l_codemf(StandoneCode, model, staqc_sql_f, hnn_lable_sql_path)
            print("第一次codemf标签已打完")
            u2l_textsa(StandoneCode, model, staqc_sql_f, hnn_lable_sql_path)
            print("第一次textsa标签已打完")
            u2l_codesa(StandoneCode, model, staqc_sql_f, hnn_lable_sql_path)
            print("第一次codesa标签已打完")

            # 标签分析
            print("标签分析开始：")
            final_analay(hnn_lable_sql_path, save_path_final_label_staqc_sql)

            # 替换hnn标签到已达标签语料中
            print("替换hnn标签到已达标签语料中：")
            final_analay_large(staqc_sql_final_label, hnn_lable_sql_path,
                               large_single_path, save_path_final_label_large_sql_mul)

        elif args.train == "python":
            # 第一次打标签
            print("第一次打标签开始：")
            u2l_codemf(StandoneCode, model, staqc_python_f,
                       hnn_lable_python_path)
            print("第一次codemf标签已打完")
            u2l_textsa(StandoneCode, model, staqc_python_f,
                       hnn_lable_python_path)
            print("第一次textsa标签已打完")
            u2l_codesa(StandoneCode, model, staqc_python_f,
                       hnn_lable_python_path)
            print("第一次codesa标签已打完")

            # 标签分析
            print("标签分析开始：")
            final_analay(hnn_lable_python_path,
                         save_path_final_label_staqc_python)

            # 替换hnn标签到已达标签语料中
            print("替换hnn标签到已达标签语料中：")
            final_analay_large(staqc_python_final_label, hnn_lable_python_path,
                               large_single_python_path, save_path_final_label_large_python_mul)

'''--------------------------------sql打标签-----------------------------------'''
# 第一次执行:codemf
StandoneCode.load_model_epoch(
    model, 86, 0.25, 0.25, 0.25, 0.25, 0.0004000000000000001)
# 第二次执行:text_sa
StandoneCode.load_model_epoch(model, 1033, 0.1, 0.1, 0.1, 0.1, 1.0002)
# 第三次执行:code_sa
StandoneCode.load_model_epoch(model, 1111, 0.1, 0.1, 0.1, 0.1, 101)

# -----------------staqc_sql------------------------
# 第一次执行
StandoneCode.u2l_codemf(model, staqc_sql_f, staqc_sql_final_label)
# 第二次执行
StandoneCode.u2l_textsa(model, staqc_sql_f, staqc_sql_final_label)
# 第三次执行
StandoneCode.u2l_codesa(model, staqc_sql_f, staqc_sql_final_label)

# -----------------large_sql------------------------
# 第一次执行
StandoneCode.u2l_codemf(model, staqc_sql_f, large_sql_final_label)
# 第二次执行
StandoneCode.u2l_textsa(model, staqc_sql_f, large_sql_final_label)
# 第三次执行
StandoneCode.u2l_codesa(model, staqc_sql_f, large_sql_final_label)

# =====================分析最终标签==============================
# staqc:抽取codemf、testsa、codesa里面标签都为1
final_analay(staqc_sql_final_label, hnn_lable_sql_path,
             save_path_final_label_staqc_sql)
# large:抽取codemf、testsa、codesa里面标签都为1，并把之前抽出的单候选合并进去
final_analay_large(large_sql_final_label, hnn_lable_sql_path,
                   large_single_sql_path, save_path_final_label_large_sql_mul)


'''--------------------------------python打标签-----------------------------------'''
# 第一次执行：codemf
StandoneCode.load_model_epoch(model, 1166, 0.5, 0.45, 0.55, 0.45, 0.0006)
# 第二次执行：text_sa
StandoneCode.load_model_epoch(model, 1079, 0.5, 0.5, 0.5, 0.5, 1.0002)
# 第三次执行code_sa
StandoneCode.load_model_epoch(model, 138, 0.15, 0.15, 0.15, 0.15, 101)

# -----------------staqc_python------------------------
# 第一次执行
StandoneCode.u2l_codemf(model, staqc_python_f, staqc_python_final_label)
# 第二次执行
StandoneCode.u2l_textsa(model, staqc_python_f, staqc_python_final_label)
# 第三次执行
StandoneCode.u2l_codesa(model, staqc_python_f, staqc_python_final_label)

# -----------------large_python------------------------
# 第一次执行
StandoneCode.u2l_codemf(model, large_python_f, large_python_final_label)
# 第二次执行
StandoneCode.u2l_textsa(model, large_python_f, large_python_final_label)
# 第三次执行
StandoneCode.u2l_codesa(model, large_python_f, large_python_final_label)

# =====================分析最终标签==============================
# staqc:抽取codemf、testsa、codesa里面标签都为1
final_analay(staqc_python_final_label, hnn_lable_python_path,
             save_path_final_label_staqc_python)
# large:抽取codemf、testsa、codesa里面标签都为1,并把之前抽出的单候选合并进去
final_analay_large(large_python_final_label, hnn_lable_python_path,
                   large_single_python_path, save_path_final_label_large_python_mul)
