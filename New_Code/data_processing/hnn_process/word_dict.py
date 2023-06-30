import pickle


def get_vocab(corpus1, corpus2):
    word_vocab = set()
    for data in corpus1:
        for item in data[1][0]:
            word_vocab.add(item)
        for item in data[1][1]:
            word_vocab.add(item)
        for item in data[2][0]:
            word_vocab.add(item)
        for item in data[3]:
            word_vocab.add(item)

    for data in corpus2:
        for item in data[1][0]:
            word_vocab.add(item)
        for item in data[1][1]:
            word_vocab.add(item)
        for item in data[2][0]:
            word_vocab.add(item)
        for item in data[3]:
            word_vocab.add(item)

    print(len(word_vocab))
    return word_vocab


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')


def vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = eval(f.read())
        f.close()

    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())
        f.close()

    word_vocab = get_vocab(total_data1, total_data2)
    with open(save_path, "w") as f:
        f.write(str(word_vocab))


def final_vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
        f.close()

    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())
        f.close()

    word_set = set()
    for data in total_data2:
        for item in data[1][0]:
            if item not in total_data1:
                word_set.add(item)
        for item in data[1][1]:
            if item not in total_data1:
                word_set.add(item)
        for item in data[2][0]:
            if item not in total_data1:
                word_set.add(item)
        for item in data[3]:
            if item not in total_data1:
                word_set.add(item)

    print(len(total_data1))
    print(len(word_set))
    with open(save_path, "w") as f:
        f.write(str(word_set))


if __name__ == "__main__":
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    vocab_processing(python_hnn, python_staqc, python_word_dict)
    vocab_processing(sql_hnn, sql_staqc, sql_word_dict)

    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    final_vocab_processing(
        python_word_dict, new_python_large, large_word_dict_python)
