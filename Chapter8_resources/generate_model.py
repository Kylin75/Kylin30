# 第8章/generate_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


# 读取数据集，并将字符编码指定为gbk，防止中文报错
penguin_df = pd.read_csv('penguins-chinese.csv', encoding='gbk')  
# 删除缺失值所在的行
penguin_df.dropna(inplace=True)  
# 将企鹅的种类定义为目标输出变量
output = penguin_df['企鹅的种类']  
# 将去除年份列（代码中未体现年份列操作，按逻辑是筛选有用特征）不作为特征列，
# 使用企鹅栖息的岛屿、喙的长度、翅膀的长度、身体质量、性别作为特征列
features = penguin_df[['企鹅栖息的岛屿', '喙的长度', '喙的深度', '翅膀的长度', '身体质量', '性别']]  
# 对特征列进行独热编码，将分类特征转为数值型，方便模型处理
features = pd.get_dummies(features)  
# 将目标输出变量转换为离散数值，把类别文本转成可用于模型训练的数值标签
output_codes, output_uniques = pd.factorize(output)  

# 从features和output_codes这两个数组中将数据集划分为训练集和测试集
# 训练集为80%，测试集为20%（1 - 80%）
# 返回的x_train和y_train为划分得到的训练集特征和标签，x_test和y_test为测试集特征和标签
# 这里标签和目标输出变量是一个意思，只是格式适配模型训练预测流程
x_train, x_test, y_train, y_test = train_test_split(features, output_codes, train_size=0.8)  

# 构建一个随机森林分类器，初始化模型对象
rfc = RandomForestClassifier()  
# 使用训练集数据x_train和y_train来拟合（训练）模型，让模型学习数据规律
rfc.fit(x_train, y_train)  
# 用训练好的模型rfc对测试集数据x_test进行预测，将预测结果存储在y_pred中
y_pred = rfc.predict(x_test)  
# 计算测试集上模型的预测准确率
# 方法是使用accuracy_score方法，比对真实标签y_test和预测标签y_pred
# 返回预测正确的样本占全部样本的比例，即准确率
score = accuracy_score(y_test, y_pred)  
# 打印模型在测试集上的准确率
print(f'该模型的准确率是：{score}')

# 返回预测正确的样本占全部样本的比例，即准确率
score = accuracy_score(y_test, y_pred)

# 使用 with 语句简化文件操作
# open() 函数和 'wb' 参数用于创建并写入字节流
# pickle.dump() 方法将模型对象转换为字节流，实现模型持久化存储
with open('rfc_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)

# 同上逻辑，将映射变量（模型输出与真实物种名称映射关系）写入文件中
with open('output_uniques.pkl', 'wb') as f:
    pickle.dump(output_uniques, f)

print('保存成功，已生成相关文件。')

