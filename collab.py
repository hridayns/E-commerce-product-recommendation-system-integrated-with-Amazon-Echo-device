import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from scipy.sparse import csr_matrix,coo_matrix
from tensorflow.contrib.factorization.python.ops import factorization_ops

#IF ONLY UID IS GIVEN, DONT USE REFINEDF() FUNCTION
#ELSE IF BOTH ARE GIVEN , CALL REFINEDF() FOR SIM MATRIX
def rec(test_uid,test_iid):

	test_uid = int(test_uid)
	dataPath = os.path.join(os.getcwd(),'datasets','CollabData1')

	productsDataFile = os.path.join(dataPath,'amazonSample1.csv')
	transactionalDataFile = os.path.join(dataPath,'transData.csv')

	t_fields = ['user_id','item_id','rating','timestamp']
	tdf = pd.read_csv(transactionalDataFile,usecols=t_fields)
	p_fields = ['unique','product','category','rating','price','description']
	pdf = pd.read_csv(productsDataFile,usecols=p_fields)
	pdf = pdf.fillna('')

	if(test_iid is not None):
		test_iid = int(test_iid)
		user_ids_that_bought_or_rated_item_i = tdf.loc[ tdf['item_id'] == test_iid,'user_id' ].values
		user_ids_that_bought_or_rated_item_i = np.append(user_ids_that_bought_or_rated_item_i,test_uid)
		tdf = tdf.loc[ tdf['user_id'].isin(user_ids_that_bought_or_rated_item_i) ]

		cate = pdf.loc[ pdf['unique'] == test_iid ,'category'].iloc[0]
		cate = cate.split('>')
		cate = cate[0]
		pdf = pdf.loc[pdf['category'].str.contains(cate)]

	user_ids = tdf['user_id'].values
	item_ids = tdf['item_id'].values
	ratings = tdf['rating'].values

	# #INVERSE = INDICES IN THE SORTED UNIQUE ARRAY IN ORDER, SO THAT IT CAN RECONSTRUCT ORIGINAL ARRAY
	# #INDEX = INDEX OF THE FIRST OCCURENCE OF THE UNIQUE ELEMENT IN ORIGINAL ARRAY (NOT USED HERE)

	uniq_uids, uniq_uid_indices, u_inv, uniq_uid_counts = np.unique(user_ids,return_index = True,return_counts = True,return_inverse=True)
	uniq_iids, uniq_iid_indices, i_inv, uniq_iid_counts, = np.unique(item_ids,return_index = True,return_counts = True,return_inverse=True)

	n_users = uniq_uids.shape[0]
	m_items = uniq_iids.shape[0]

	sim_matrix = np.zeros((n_users,m_items))

	csr_ratings = csr_matrix((ratings,(user_ids,item_ids)))
	sim_matrix[u_inv,i_inv] = csr_ratings[uniq_uids[u_inv],uniq_iids[i_inv]]
	coo_ratings = coo_matrix(sim_matrix)



	# cate = pdf.loc[ pdf['unique'] == test_iid ,'category'].iloc[0]
	# cate = cate.split('>')
	# cate = cate[0]
	# pdf = pdf.loc[pdf['category'].str.contains(cate)]

	# tensor_indices = np.column_stack((coo_train.row,coo_train.col))
	tensor_indices = np.column_stack((coo_ratings.row,coo_ratings.col))

	k = 34 #latent factors
	num_iterations = 20

	# input_tensor = tf.SparseTensor(indices=tensor_indices,values = (coo_train.data).astype(np.float32),dense_shape=coo_train.shape)
	input_tensor = tf.SparseTensor(indices=tensor_indices,values = (coo_ratings.data).astype(np.float32),dense_shape=coo_ratings.shape)
	# model = factorization_ops.WALSModel(coo_train.shape[0], coo_train.shape[1], k)
	model = factorization_ops.WALSModel(coo_ratings.shape[0], coo_ratings.shape[1], k)
	row_factor = model.row_factors[0]
	col_factor = model.col_factors[0]
	row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
	col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

	with tf.Session() as sess:
		sess.run(model.initialize_op)
		sess.run(model.worker_init)
		for _ in range(num_iterations):
			sess.run(model.row_update_prep_gramian_op)
			sess.run(model.initialize_row_update_op)
			sess.run(row_update_op)
			sess.run(model.col_update_prep_gramian_op)
			sess.run(model.initialize_col_update_op)
			sess.run(col_update_op)
		output_row = row_factor.eval(session=sess)
		output_col = col_factor.eval(session=sess)

	pred_mat = csr_matrix(np.matmul(output_row,output_col.T)).toarray()


	for u in range(n_users):
		if(uniq_uids[u] == test_uid):
			recs = pred_mat[u]
			break

	item_id_indices_alr_rated_by_u = sim_matrix[u].nonzero()[0]
	recs[item_id_indices_alr_rated_by_u] = -1

	rec_indices = np.argsort(recs)
	rec_indices_desc = np.flip(rec_indices,0) 
	recommended_iids = uniq_iids[rec_indices_desc]
	recommended_iids = recommended_iids[0:10]

	frames = []
	for x in recommended_iids:
		frames.append(pdf.loc[pdf['unique'] == x])

	rec_df = pd.concat(frames)
	rec_df = rec_df.loc[:,['unique','product','category','price','description']]
	result = rec_df.to_dict('records')
	return result
	# rec_df = pdf.loc[ pdf['unique'].isin(recommended_iids),['unique','product','category','price','description'] ]
	# top_recs = rec_df[0:10]

	# dictOutput = rec_df[0:10].to_dict('records')
	# dictOutput = top_recs.to_dict('records')
	# print(dictOutput)
	# return dictOutput
# print(top_recs.to_dict())
# print(top_recs.head(3))

# for x in range(top_recs.shape[0]):
        # print('Product : ',top_recs['product'].iloc[x],"\nCategory: ",top_recs['category'].iloc[x],'\n\n')