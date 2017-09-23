import sys
import convert

if (sys.argv[1] == 'save'):
    fname = '../UD_Indonesian_Conll2017/id-ud-train.conllu'
    fsave = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
    list_complete = convert.save_list(fname, fsave)
elif (sys.argv[1] == 'load'):
    fload = '../UD_Indonesian_Conll2017/list_of_tuple.pickle'
    list_complete = convert.load_list(fload)
