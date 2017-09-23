import pickle

def save_list(fname, fsave):
    # fname = '../UD_Indonesian_Conll2017/id-ud-train.conllu'

    with open(fname) as f:
        content = f.readlines()
    # content = [x.strip() for x in content]

    list_complete = []
    list_of_tuple = None

    for line in content:
    	if line[0] != '\n' and line[0] != '#':
    		if line[0:2] == '1\t':
    			list_complete.append(list_of_tuple)
    			list_of_tuple = []
    		tokens = line.split('\t')
    		list_of_tuple.append((tokens[1],tokens[3]))

    list_complete = list_complete[1:]

    print(list_complete[0])
    print('--------------')
    print(list_complete[1])

    # fsave = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
    with open(fsave, 'wb') as handle:
        pickle.dump(list_complete, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return list_complete

def load_list(fload):
    # How to load file
    # fload = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
    with open(fload, 'rb') as handle:
        list_complete = pickle.load(handle)

    print(list_complete[0])
    print('--------------')
    print(list_complete[1])

    return list_complete

# # Check result
# print(list_complete == loaded)
