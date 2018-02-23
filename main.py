import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

print('初始化路径和变量...')
names = ['user_id', 'item_id', 'rating', 'timestamp']
trainingset_file = 'data/u3.base'
testset_file = 'data/u3.test'
n_users = 943
n_items = 1682
ratings = np.zeros((n_users, n_items))
df = pd.read_csv(trainingset_file, sep='\t', names=names)
print('loading training file...')
print('the top 5 sample of the file')
print(df.head())
for row in df.itertuples():
	ratings[row[1] - 1, row[2] - 1] = row[3]
print('scale of rating matrix: %d * %d.' % (n_users, n_items))
print('num of valid data in traing set is %d.' % len(df))

# 计算矩阵密度
def cal_sparsity():
	sparsity = float(len(ratings.nonzero()[0]))
	sparsity /= (ratings.shape[0] * ratings.shape[1])
	sparsity *= 100
	print('the sparsity of training set is {:4.2f}%'.format(sparsity))
print(cal_sparsity())

def rmse(pred, actual):
	from sklearn.metrics import mean_squared_error
	pred = pred[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()]. flatten()
	return np.sqrt(mean_squared_error(pred, actual))

print('--------baseline----------')

def cal_mean():
	'''计算一些均值'''
	print('计算训练集各项统计数据...')
	print('计算总体均值，各user打分均值，各item打分均值...\n请稍后...')
	global all_mean, user_mean, item_mean
	all_mean = np.mean(ratings[ratings != 0])
	user_mean = sum(ratings.T) / sum((ratings != 0). T)
	item_mean = sum(ratings) / sum((ratings != 0))

	user_mean_nan = '是'
	item_mean_nan = '是'
	if np.isnan(user_mean).any():
		user_mean_nan = '否'
	if np.isnan(item_mean).any():
		item_mean_nan = '否'
	print('是否存在User均值为NaN?', user_mean_nan)
	print('是否存在Item均值为NaN?', item_mean_nan)
	print('对NaN填充总体均值...')

	user_mean = np.where(np.isnan(user_mean), all_mean, user_mean)
	item_mean = np.where(np.isnan(item_mean), all_mean, item_mean)
	if np.isnan(user_mean).any():
		user_mean_nan = '否'
	if np.isnan(item_mean).any():
		item_mean_nan = '否'
	print('是否存在User均值为NaN?', user_mean_nan)
	print('是否存在Item均值为NaN?', item_mean_nan)
	print('均值计算完成，总体打分均值为 %.4f' % all_mean)

cal_mean()

def predict_naive(user, item):
	prediction = item_mean[item] + user_mean[user] - all_mean
	return prediction

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
print(test_df.head())
predictions = []
targets = []
print('the scale of test set is %d' % len(test_df))
print('using baseline algorithm to predict...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_naive(user, item))
	targets.append(actual)

print('the rmse of test result is %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------item-based协同过滤算法（相似度未归一化）------')

def cal_similarity(ratings, kind, epsilon=1e-9):
	'''利用余弦距离计算相似度'''
	'''epsilon 防止分母为0的异常'''
	if kind == 'user':
		sim = ratings.dot(ratings.T) + epsilon
	elif kind == 'item':
		sim = ratings.T.dot(ratings) + epsilon
	norms = np.array([np.sqrt(np.diagonal(sim))])
	return (sim / norms / norms.T)

print('计算相似度矩阵...')
user_similarity = cal_similarity(ratings, kind='user')
item_similarity = cal_similarity(ratings, kind='item')
print('计算完成')
print('相似度矩阵样例： （item-item）')
print(np.round_(item_similarity[:10, :10], 3))

def predict_itemCF(user, item, k=100):
	'''item-based协同过滤算法，预测rating'''
	nzero = ratings[user].nonzero()[0]
	prediction = ratings[user, nzero].dot(item_similarity[item, nzero])\
					/ sum(item_similarity[item,nzero])
	return prediction 

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using item-based cf algorithm to do the prediction...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_itemCF(user, item))
	targets.append(actual)

print('the rmse of test set is %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------结合基线算法的item-based协同过滤算法（相似度未归一化）------')

def predict_itemCF_baseline(user, item, k=100):
	nzero = ratings[user].nonzero()[0]
	baseline = item_mean + user_mean[user] - all_mean
	prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero])\
                / sum(item_similarity[item, nzero]) + baseline[item]
	return prediction

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using item-based cf algorithm with baseline algorithm to do the prediction...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_itemCF_baseline(user, item))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------user-based协同过滤算法（相似度未归一化）------')

def predict_userCF(user, item, k=10):
	nzero = ratings[user].nonzero()[0]
	baseline = user_mean + item_mean[item] - all_mean
	prediction = ratings[nzero, item].dot(user_similarity[user, nzero]) \
					/ sum(user_similarity[user, nzero])
	# 冷启动问题，该item暂时没有评分
	if np.isnan(prediction):
		prediction = baseline[user]
	return prediction

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using user-based cf algorithm to do the prediction...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_userCF(user, item))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------the combination of user-based cf algorithm and baseline algorithm------')

def predict_usercf_baseline(user, item, k=10):
	nzero = ratings[:, item].nonzero()[0]
	baseline = item_mean[item] + user_mean - all_mean
	prediction = (ratings[nzero, item] - baseline[nzero]).dot(user_similarity[user, nzero]) \
					/ sum(user_similarity[user, nzero]) + baseline[user]
	if np.isnan(prediction):
		prediction = baseline[user]
	return prediction

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using user-based cf algorithm with baseline algorithm to do the prediction...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_usercf_baseline(user, item))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------item-based cf alg with baseline------')

def predict_biasCF(user, item, k=100):
	nzero = ratings[user].nonzero()[0]
	baseline = item_mean + user_mean[user] - all_mean
	prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero]) \
						/ sum(item_similarity[item, nzero]) + baseline[item]
	if prediction > 5:
		prediction = 5
	if prediction < 1:
		prediction = 1 
	return prediction 

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using item-based cf algorithm with baseline algorithm to do the prediction...')
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_biasCF(user, item))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------top-k cf(item-based + baseline)------')

def predict_topkCF(user, item, k=10):
	nzero = ratings[user].nonzero()[0]
	baseline = item_mean + user_mean[user] - all_mean
	choice = nzero[item_similarity[item, nzero].argsort()[::-1][:k]]
	prediction = (ratings[user, choice] - baseline[choice]).dot(item_similarity[item, choice]) \
						/ sum(item_similarity[item, choice]) + baseline[item]
	if prediction > 5:
		prediction = 5
	if prediction < 1:
		prediction = 1 
	return prediction 

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions=[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using item-based cf algorithm with baseline algorithm to do the prediction...')
k = 20
print('the k is %d' % k)
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_topkCF(user, item, k))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))

print('------baseline + item-based + topk + normalized matrix------')

def cal_similarity_norm(ratings, kind, epsilon=1e-9):
	if kind == 'user':
		# 对同一个user的打分归一化
		rating_user_diff = ratings.copy()
		for i in range(ratings.shape[0]):
			nzero = ratings[i].nonzero()
			rating_user_diff[i][nzero] = ratings[i][nzero] - user_mean[i]
		sim = rating_user_diff.dot(rating_user_diff.T) + epsilon
	elif kind == 'item':
		# 对同一个item的打分归一化
		rating_item_diff = ratings.copy()
		for j in range(ratings.shape[1]):
			nzero = ratings[:,j].nonzero()
			rating_item_diff[:,j][nzero] = ratings[:,j][nzero] - item_mean[j]
		sim = rating_item_diff.dot(rating_item_diff.T) + epsilon
	norms = np.array([np.sqrt(np.diagonal(sim))])
	return (sim / norms / norms.T)

print('计算归一化的相似度矩阵...')
user_similarity_norm = cal_similarity_norm(ratings, kind='user')
item_similarity_norm = cal_similarity_norm(ratings, kind='item')
print('calculation finished')
print('example of similarity matrix: (item-item)')
print(np.round_(item_similarity_norm[:10, :10], 3))

def predict_norm_cf(user, item, k=20):
	nzero = ratings[user].nonzero()[0]
	baseline = item_mean + user_mean[user] - all_mean
	choice = nzero[item_similarity_norm[item, nzero].argsort()[::-1][:k]]
	prediction = (ratings[user, choice] - baseline[choice]).dot(item_similarity_norm[item, choice]) \
					/ sum(item_similarity_norm[item, choice]) + baseline[item]
	if prediction > 5:
		prediction = 5
	if prediction < 1:
		prediction = 1 
	return prediction 

print('loading test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions =[]
targets=[]
print('scale of test set is %d' % len(test_df))
print('using baseline + item-based + topk + normalized matrix to do the prediction...')
k = 20
print('the k is %d' % k)
for row in test_df.itertuples():
	user, item, actual = row[1] - 1, row[2] - 1, row[3]
	predictions.append(predict_norm_cf(user, item, k))
	targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
