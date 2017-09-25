
import pickle

pkl_file = 'protein_nomination_value.pkl'
money = pickle.load(open(pkl_file, "rb"))


test_df = pickle.load(open('putative_set.pkl', "rb"))
training_df = pickle.load(open('training_set.pkl', "rb"))

winners = test_df[test_df['id'].isin(money.keys())]


losers = test_df[~test_df['id'].isin(money.keys())]
test_df['label'] =0
test_df.set_value(test_df['id'].isin(money.keys()), 'label', 1)



test_df.corr()['label'][test_df.corr()['label'] > 0.8]


winners['count_M', 'count_N', 'count_C']


