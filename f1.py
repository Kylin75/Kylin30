import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="企鹅分类器",  # 页面标题
    page_icon=":penguin:",  # 页面图标（Streamlit 支持的 emoji 语法）
    layout="wide",  # 页面布局：宽屏模式
)

# 使用侧边栏实现多页面显示效果
with st.sidebar:
    # 加载并显示侧边栏 Logo（相对路径，需确保 images 目录存在对应文件）
    st.image('Chapter8_resources\images/rigth_logo.png', width=100)  
    st.title('请选择页面')  # 侧边栏标题
    # 创建页面选择框，隐藏默认标签（label_visibility='collapsed'）
    page = st.selectbox("请选择页面", ["简介页面", "预测分类页面"], label_visibility='collapsed')  

# 根据选择的页面，显示不同内容
if page == "简介页面":
    st.title("企鹅分类器:penguin:")  # 主内容区标题
    st.header('数据集介绍')  # 二级标题
    # 用 markdown 写数据集说明
    st.markdown("""帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，也可以作为机器学习入门练习。  
    该数据集是由 Gorman 等收集，并发布在一个名为 palmerpenguins 的 R 语言包，以对南极企鹅种类进行分类和研究。  
    该数据集记录了 344 行观测数据，包含 3 个不同物种的企鹅：阿德利企鹅、巴布亚企鹅和帽带企鹅的各种信息。""")
    st.header('三种企鹅的卡通图像')  # 二级标题
    # 加载并显示企鹅卡通图（相对路径）
    st.image('Chapter8_resources\images/penguins.png')  

elif page == "预测分类页面":
    st.header("预测企鹅分类")  # 主内容区标题
    # 用 markdown 写应用说明
    st.markdown("这个 Web 应用是基于帕尔默群岛企鹅数据集构建的模型，只需输入 6 个信息，就可以预测企鹅的物种，使用下面的表单开始预测吧！")  

    # 该页面是 3:1:2 的列布局：用 st.columns 划分三列，比例 [3,1,2]
    col_form, col, col_logo = st.columns([3, 1, 2])  

    # 在第一列（col_form）里放表单
    with col_form:
        # 用 st.form 创建表单，key 为 'user_inputs'，表单内组件需点提交才会触发逻辑
        with st.form('user_inputs'):  
            # 岛屿选择框：options 是可选值，label 显示“企鹅栖息的岛屿”
            island = st.selectbox('企鹅栖息的岛屿', options=['托尔多岛', '比斯科群岛', '德里姆岛'])  
            # 性别选择框：options 是可选值，label 显示“性别”
            sex = st.selectbox('性别', options=['雄性', '雌性'])  
            # 喙的长度输入框：数字输入，最小值 0.0，label 显示“喙的长度（毫米）”
            bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)  
            # 喙的深度输入框：数字输入，最小值 0.0，label 显示“喙的深度（毫米）”
            bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)  
            # 翅膀长度输入框：数字输入，最小值 0.0，label 显示“翅膀的长度（毫米）”
            flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)  
            # 身体质量输入框：数字输入，最小值 0.0，label 显示“身体质量（克）”
            body_mass = st.number_input('身体质量（克）', min_value=0.0)  
            # 表单提交按钮，点击后表单内数据会被提交
            submitted = st.form_submit_button('预测分类')  

    # 初始化数据预处理中与岛屿相关的变量（独热编码用，默认全 0）
    island_biscoe, island_dream, island_torgerson = 0, 0, 0  
    # 根据用户选的岛屿，更新对应独热编码变量
    if island == '比斯科群岛':
        island_biscoe = 1
    elif island == '德里姆岛':
        island_dream = 1
    elif island == '托尔多岛':
        island_torgerson = 1

    # 初始化数据预处理中与性别相关的变量（独热编码用，默认全 0）
    sex_female, sex_male = 0, 0  
    # 根据用户选的性别，更新对应独热编码变量
    if sex == '雌性':
        sex_female = 1
    elif sex == '雄性':
        sex_male = 1

    # 把用户输入的特征 + 编码后的变量，整理成模型需要的格式
    format_data = [bill_length, bill_depth, flipper_length, body_mass,
                   island_dream, island_torgerson, island_biscoe, sex_male, sex_female]

    # 从本地加载训练好的随机森林模型（rb：二进制读模式）
    with open('Chapter8_resources\rfc_model.pkl', 'rb') as f:  
        rfc_model = pickle.load(f)

    # 从本地加载“类别编码映射”文件（用于把模型输出的编码转成企鹅物种名）
    with open('Chapter8_resources\output_uniques.pkl', 'rb') as f:  
        output_uniques_map = pickle.load(f)

    # 如果用户点击了“预测分类”按钮，执行预测逻辑
    if submitted:
        # 用 pd.DataFrame 把输入数据转成模型训练时的格式（列名要和模型特征名一致）
        format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)  
        # 用加载的模型预测，得到物种编码
        predict_result_code = rfc_model.predict(format_data_df)  
        # 通过映射关系，把编码转成具体的物种名称
        predict_result_species = output_uniques_map[predict_result_code][0]  

        # 在页面显示预测结果（用 markdown 语法加粗）
        st.write(f'根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**')  

    # 在第三列（col_logo）里根据状态显示不同图片
    with col_logo:
        # 如果没提交表单，显示默认 Logo
        if not submitted:  
            st.image('Chapter8_resources\images/rigthe_logo.png', width=300)
        # 如果提交了表单，显示预测物种对应的图片（假设 images 目录有对应物种名的图片）
        else:  
            st.image(f'Chapter8_resources\images/{predict_result_species}.png', width=300)
