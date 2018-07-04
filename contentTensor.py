import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

dataPath = os.path.join(os.getcwd(),'datasets','CollabData1')

productsDataFile = os.path.join(dataPath,'amazonSample1.csv')

df = pd.read_csv(productsDataFile, low_memory=False, encoding='ISO-8859-1')

df = df[['unique','product','brand','category','description']]
df = df.fillna('')

def recommend(item):
	print("For product", item , ":")
	item_df = df[df['product'].str.contains(item)].drop_duplicates()
	cate = item_df['category'].iloc[0]
	# print(type(cate))
	new_brand = item_df['brand'].iloc[0]
	cate_array = str(cate).split('>')
	if len(cate_array) > 3:
		sub_cate = cate_array[0:-2]
		new_cate = '>'.join(sub_cate)
	elif len(cate_array) >1:
		new_cate = cate_array[0] + ">" + cate_array[1]
	else:
		new_cate = cate_array[0]
	
	new_df = df[df['category'].str.contains(new_cate)].drop_duplicates()
	
	tfidf = TfidfVectorizer(stop_words='english')
	
	matrix = tfidf.fit_transform(new_df['description'])
	matrix  = matrix.toarray()
	
	
	#TensorFlow
	a = tf.placeholder(tf.float32, shape=matrix.shape, name = 'a')
	b = tf.placeholder(tf.float32, shape=matrix.T.shape, name = 'b')
	normal_a = tf.nn.l2_normalize(a,0)
	normal_b = tf.nn.l2_normalize(b,0)
	cos_sim = (tf.matmul(normal_a, normal_b))
	sess = tf.Session()
	cos_score = sess.run(cos_sim, feed_dict={a:matrix, b:matrix.T}) 
	sess.close()
	
	indices = pd.Series(new_df.index, index=new_df['product']).drop_duplicates()
	
	idx = indices[item]
	
	sim_scores = list(enumerate(cos_score[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	item_indices = [i[0] for i in sim_scores]
	new_df = new_df.iloc[item_indices]
	
	df_brand = new_df.loc[new_df['brand'] == new_brand]
	df_non_brand = new_df.loc[new_df['brand'] != new_brand]
	
	return df_brand[:5], df_non_brand[:5]
	
u3_brand, u3_non_brand = recommend('Hornby 2014 Catalogue')
print("\n The recommended products are:\n\n")
print("Similar product by the same brand\n")
for i in range(u3_brand.shape[0]):
	print("Product: ",u3_brand['product'].iloc[i])
print("\nSimilar Products\n")
for i in range(u3_non_brand.shape[0]):
	print('Product: ',u3_non_brand['product'].iloc[i])

#os.remove('output.csv')
#u3.to_csv('output.csv')