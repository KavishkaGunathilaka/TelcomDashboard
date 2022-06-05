import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import lightgbm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Dashboard", 
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": None,
        "Report a Bug":None,
        "About":None
    }
)

if 'primaryColor' not in st.session_state:
    st.session_state['primaryColor'] = '#04B59B'
if 'secondaryColor' not in st.session_state:
    st.session_state['secondaryColor'] = '#01301f'
if 'colorScale' not in st.session_state:
    st.session_state['colorScale'] = 'blugrn'

graphWidth = 400
graphHeight = 400
graphMargins = dict(l=0, r=0, t=23, b=0)

@st.cache
def readData(fileName):
    return pd.read_csv(fileName)

@st.cache
def readModel(fileName):
    return joblib.load(fileName)

def removeOutliers(df, column, tol):
    q1 = df[column].quantile(q=0.25)
    q3 =  df[column].quantile(q=0.75)
    IQRVal = (q3 -q1)*tol
    minVal =  q1 - IQRVal
    maxVal = q3 + IQRVal
    df.loc[(df[column] < minVal) | (df[column] > maxVal), column] = np.nan

def labelEncode(df):
    le = LabelEncoder()
    df.voice_mail_plan = le.fit_transform(df.voice_mail_plan)
    df.intertiol_plan = le.fit_transform(df.intertiol_plan)
    df.location_code = le.fit_transform(df.location_code)
    return df

def preprocess(dataFrame):
    dataFrame.drop(['Unnamed: 19', 'Unnamed: 20'], axis = 1, inplace=True, errors='ignore')

    dataFrame = dataFrame.drop_duplicates()

    dataFrame.loc[(dataFrame.number_vm_messages < 0), 'number_vm_messages'] = np.nan
    dataFrame.loc[(dataFrame.total_day_calls < 0), 'total_day_calls'] = np.nan
    dataFrame.loc[(dataFrame.total_day_charge < 0), 'total_day_charge'] = np.nan
    dataFrame.loc[(dataFrame.total_day_min < 0), 'total_day_min'] = np.nan
    dataFrame.loc[(dataFrame.total_eve_calls < 0), 'total_eve_calls'] = np.nan
    dataFrame.loc[(dataFrame.total_eve_min < 0), 'total_eve_min'] = np.nan
    dataFrame.loc[(dataFrame.total_intl_calls < 0), 'total_intl_calls'] = np.nan
    dataFrame.loc[(dataFrame.total_intl_minutes < 0), 'total_intl_minutes'] = np.nan
    dataFrame.loc[(dataFrame.total_night_minutes < 0), 'total_night_minutes'] = np.nan

    removeOutliers(dataFrame, 'number_vm_messages', 1.5)
    removeOutliers(dataFrame, 'total_day_calls', 1.5)
    removeOutliers(dataFrame, 'total_day_min', 1.5)
    removeOutliers(dataFrame, 'total_eve_min', 1.5)
    removeOutliers(dataFrame, 'total_night_minutes', 1.5)
    removeOutliers(dataFrame, 'total_night_charge', 1.5)

    # Imputing missing values ------------------------------------------------------
    imputer = KNNImputer(n_neighbors=3)

    temp = dataFrame.drop(['voice_mail_plan', 'intertiol_plan', 'location_code'], axis=1)
    df_imputed = pd.DataFrame(imputer.fit_transform(temp), columns = temp.columns)

    df_imputed[['voice_mail_plan', 'intertiol_plan', 'location_code']] =  dataFrame[['voice_mail_plan', 'intertiol_plan', 'location_code']]
    dataFrame = df_imputed

    dataFrame.voice_mail_plan.fillna(dataFrame.voice_mail_plan.mode()[0], inplace=True)
    dataFrame.intertiol_plan.fillna(dataFrame.intertiol_plan.mode()[0], inplace=True)
    dataFrame.location_code.fillna(dataFrame.location_code.mode()[0], inplace=True)

    dataFrame = labelEncode(dataFrame)

    dataFrame['total_mins'] = dataFrame['total_day_min'] + dataFrame['total_eve_min'] + dataFrame['total_night_minutes'] + dataFrame['total_intl_minutes']
    dataFrame['total_calls'] = dataFrame['total_day_calls'] + dataFrame['total_eve_calls'] + dataFrame['total_night_calls'] + dataFrame['total_intl_calls']
    dataFrame['total_charge'] = dataFrame['total_day_charge'] + dataFrame['total_eve_charge'] + dataFrame['total_night_charge'] + dataFrame['total_intl_charge']
    dataFrame['mins_per_call'] = dataFrame['total_mins']/dataFrame['total_calls']
    dataFrame['has_plan'] = dataFrame['intertiol_plan'] | dataFrame['voice_mail_plan']

    return dataFrame

def catSummary(column):
    mode = column.mode()
    return mode[0]
    

df = readData("data_str.csv")
df_enc = readData("data.csv")
feature_importances = readData("feature_importances.csv")
model = readModel("lgb.pkl")
catFeatures = ['voice_mail_plan', 'intertiol_plan', 'location_code', 'Churn', 'has_plan']


with st.sidebar:
    sideCol1, sideCol2, sideCol3 = st.columns(3)
    with sideCol1:
        primaryColor = st.color_picker('Primary Color', '#04B59B')
        st.session_state['primaryColor'] = primaryColor
    with sideCol2:
        secondaryColor = st.color_picker('Secondary Color', '#E6EA09')
        st.session_state['secondaryColor'] = secondaryColor
    with sideCol3:
        colorScale = st.selectbox('Color scheme', options = px.colors.named_colorscales(), index = 6)
        st.session_state['colorScale'] = colorScale
    st.markdown("""---""")
    colsToPlot = st.multiselect('Pairplot: Select features', options = list(df_enc.columns), default = ['total_day_calls', 'total_eve_calls'], key=0)
    st.markdown("""---""")
    colToPlot = st.selectbox('Histogram: Selected features', options = list(df.columns), index = 5)
    distType = st.radio("Distribution type", ('box', 'violin', 'rug'))
    st.markdown("""---""")
    colsToCorr = st.multiselect('Correlations: Selected features', options = list(df_enc.columns), default = ['total_day_calls', 'total_eve_calls', 'mins_per_call', 'total_night_calls'], key=1)
    st.markdown("""---""")

st.markdown("<h1 style='text-align: center; color:" + str(st.session_state['primaryColor']) + "';>Telcom Dashboard</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with  st.container():
        fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]], subplot_titles=('Churn', 'Voice Mail Plan', 'International Plan', 'Location'))
        fig.add_trace(go.Pie(values=df.Churn.value_counts().values, labels=['No', 'Yes'], name="Churn"), 1, 1)
        fig.add_trace(go.Pie(values=df.voice_mail_plan.value_counts().values, labels=['No', 'Yes'], name="VM plan"), 1, 2)
        fig.add_trace(go.Pie(values=df.intertiol_plan.value_counts().values, labels=['No', 'Yes'], name="Intnl Plan"), 2, 1)
        fig.add_trace(go.Pie(values=df.location_code.value_counts().values, labels=['452', '445', '547'], name="Location Code"), 2, 2)

        fig.update_layout(autosize=False, width=graphWidth, height=graphHeight, margin=graphMargins)
        fig.update_traces(marker=dict(colors=[st.session_state['primaryColor'], st.session_state['secondaryColor']]))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""---""")

    with st.container():
        
        plotDF = df_enc[colsToCorr]
        corr = plotDF.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale=st.session_state['colorScale'], width=400, height=400)
        fig.update_layout(margin=graphMargins)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""---""")

with col2:
    with  st.container():
        try:
            if colToPlot in catFeatures:
                fig = px.histogram(df, x=colToPlot, color="Churn",
                                    color_discrete_sequence=[st.session_state['primaryColor'], st.session_state['secondaryColor']],
                                    category_orders=dict(colToPlot=df[colToPlot].unique().tolist()),
                                    width=graphWidth, height=graphHeight
                )
                fig.update_layout(title=colToPlot, xaxis_type='category',  margin=graphMargins)
            else:
                fig = px.histogram(df, x=colToPlot, color="Churn",
                                    marginal=distType,
                                    color_discrete_sequence=[st.session_state['primaryColor'], st.session_state['secondaryColor']],
                                    width=graphWidth, height=graphHeight
                                )
                fig.update_layout(title=colToPlot, margin=graphMargins)
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        st.markdown("""---""")

    with st.container():
        try:
            fig = go.Figure(data=go.Splom(
                            dimensions=[dict(label=cl, values=df[cl]) for cl in colsToPlot],
                            marker=dict(color=df_enc['Churn'],
                                        size=5,
                                        colorscale=st.session_state['colorScale'],
                                        line=dict(width=0.5)),
                                        text = df['Churn']))
            fig.update_layout(
                            dragmode='select',
                            width=400,
                            height=400,
                            hovermode='closest',
                            margin=graphMargins)
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        st.markdown("""---""")

with col3:
    with st.container():
        fig = px.bar(feature_importances, y='value', x='feature', text_auto='.2s', title="Feature importances", labels={
                     "value": "Mean decrease in impurity",
                     "feature": "Feature Name"
                 }, width=graphWidth, height=graphHeight,  color_discrete_sequence=[st.session_state['primaryColor'], st.session_state['secondaryColor']])
        fig.update_layout(margin=graphMargins)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""---""")

    with st.container():
        modelStatLabels = ["Accuracy", 'F1 Score', 'Precision', 'Recall']
        modelStats = [0.9719827586206896, 0.9422222222222223, 0.9636363636363636, 0.9217391304347826]
        fig = px.bar(x=modelStatLabels, y=modelStats, labels={
                     'x': "Metric",
                     "y": "Value"
                 }, width=300, height=300,  color_discrete_sequence=[st.session_state['primaryColor'], st.session_state['secondaryColor']])
        fig.update_layout(title="LGBM Model Stats", margin=graphMargins)
        st.plotly_chart(fig, use_container_width=True)

        uploaded_file = st.file_uploader("Choose a csv file to predict")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data = preprocess(data)
            prediction = model.predict(data.drop(['location_code', 'customer_id'], axis=1))
            outDF = pd.DataFrame()
            outDF['customer_id'] = data['customer_id'].astype(int)
            outDF['Churn'] = prediction
            with st.expander("View predictions"):
                st.download_button("Download CSV", outDF.to_csv(index=False))
                st.dataframe(outDF)
