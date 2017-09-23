import pickle

fname = '../UD_Indonesian_Conll2017/id-ud-train.conllu'

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
print(list_complete)

with open('../UD_Indonesian_Conll2017/list_of_tuple.pickle', 'wb') as handle:
    pickle.dump(list_complete, handle, protocol=pickle.HIGHEST_PROTOCOL)

# How to load file
with open('../UD_Indonesian_Conll2017/list_of_tuple.pickle', 'rb') as handle:
    loaded = pickle.load(handle)


# Check result
print(list_complete == loaded)