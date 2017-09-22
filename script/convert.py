import pickle

fname = '../UD_Indonesian_Conll2017/id-ud-train.conllu'

with open(fname) as f:
    content = f.readlines()
# content = [x.strip() for x in content] 

list_of_tuple = []

for line in content:
	if line[0] != '\n' and line[0] != '#':
		tokens = line.split('\t')
		list_of_tuple.append((tokens[1],tokens[3]))

print(list_of_tuple)

with open('../UD_Indonesian_Conll2017/list_of_tuple.pickle', 'wb') as handle:
    pickle.dump(list_of_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)

# How to load file
with open('../UD_Indonesian_Conll2017/list_of_tuple.pickle', 'rb') as handle:
    loaded = pickle.load(handle)


# Check result
print(list_of_tuple == loaded)