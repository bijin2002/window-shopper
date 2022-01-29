import streamlit as st
import pandas as pd
import numpy as np 
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import altair as alt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from bokeh.plotting import figure




header=st.container()
dataset=st.container()
features=st.container()
modelTraning=st.container()

with header:
	st.title("Window Shopper")
	st.header("This is an app that predicts if the shopper is a Buyer or just another window shopper")
with dataset:
	header_pic = Image.open('ws.jpg')
	st.sidebar.image(header_pic, use_column_width=True)
	
    
	st.sidebar.title("Select Visual Charts")
	st.sidebar.markdown("Select the Charts/Plots accordingly:")
    
	#st.header("dataset:")
	shopping_base=pd.read_csv("online_shoppers_intention.csv")
	#st.write(shopping_base.head(20))
	st.header("Cleaned Dataset:")
	shop = shopping_base.replace({'VisitorType' : { 'New_Visitor' : 0, 'Returning_Visitor' : 1, 'Other' : 2 }})
	monthlist = shop['Month'].replace('June', 'Jun')
	mlist=[]
m = np.array(monthlist)
for i in m:
    a = list(calendar.month_abbr).index(i)
    mlist.append(a)

shop['Month'] =  mlist
shop.dropna(inplace=True)
st.write(shop.head()) 

with features:
	x=np.array(shop["Revenue"].value_counts())
mylabels=["False","True"]
fig = go.Figure(
    go.Pie(
    labels = mylabels,
    values =x ,
    hoverinfo = "label+percent",
    textinfo = "value",insidetextorientation='radial')
)

st.header("Revenue Distribution:")
st.plotly_chart(fig) 

####feature importance#######
def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)

def randomForest(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    st.subheader('Random Forest Classifier:')
    impPlot(feat_importances, 'Random Forest Classifier')
    #st.write(feat_importances)
    st.write('\n')


x = shop.iloc[:, :-1]  # Using all column except for the last column as X
y = shop.iloc[:, -1]  # Selecting the last column as Y
randomForest(x, y)

#########################################################


######charts for different variables#########
st.header("graphs")
  
chart_visual = st.sidebar.selectbox('Select Charts/Plot type', 
                                    ('Line Chart', 'Bar Chart', 'Bubble Chart'))
  
st.sidebar.checkbox("showing analysis by buyers and non buyers", True, key = 1)
selected_status = st.sidebar.selectbox('Select Status',
                                       options = ["BounceRates","Month","PageValues","SpecialDay","Region","VisitorType"])
  
fig = go.Figure()
if chart_visual == 'Line Chart':
    if selected_status == 'BounceRates':
        fig.add_trace(go.Scatter(x = shop.Revenue, y = shop.BounceRates,mode="lines",name = 'BounceRates'))
    if selected_status == 'Month':
        fig.add_trace(go.Scatter(x = shop.Month, y = shop.Revenue,mode="lines",name = 'Month'))
    if selected_status == 'PageValues':
        fig.add_trace(go.Scatter(x = shop.Revenue, y = shop.PageValues,mode="lines",name = 'PageValues'))    
    if selected_status == 'SpecialDay':
        fig.add_trace(go.Scatter(x = shop.SpecialDay, y = shop.Revenue,mode="lines",name = 'SpecialDay'))
    if selected_status == 'Region':
        fig.add_trace(go.Scatter(x = shop.Revenue, y = shop.Region,mode="lines",name = 'Region'))
    if selected_status == 'VisitorType':
        fig.add_trace(go.Scatter(x = shop.VisitorType, y = shop.Revenue,mode="lines",name = "VisitorType"))  

    
  
if chart_visual == 'Bar Chart':
    if selected_status == 'BounceRates':
        fig.add_trace(go.Bar(x = shop.Revenue, y = shop.BounceRates,name = 'BounceRates'))
    if selected_status == 'Month':
        fig.add_trace(go.Bar(x = shop.Month, y = shop.Revenue,name = 'Month'))
    if selected_status == 'PageValues':
        fig.add_trace(go.Bar(x = shop.Revenue, y = shop.PageValues,name = 'PageValues'))    
    if selected_status == 'SpecialDay':
        fig.add_trace(go.Bar(x = shop.SpecialDay, y = shop.Revenue,name = 'SpecialDay'))
    if selected_status == 'Region':
        fig.add_trace(go.Bar(x = shop.Revenue, y = shop.Region,name = 'Region'))
    if selected_status == 'VisitorType':
        fig.add_trace(go.Bar(x = shop.VisitorType, y = shop.Revenue,name = "VisitorType"))  

st.plotly_chart(fig, use_container_width=True)

if chart_visual == 'Bubble Chart':
    if selected_status == 'Month':
        fig.add_trace(go.Scatter(x = shop.BounceRates, y = shop.ExitRates,
                                 mode='markers', 
                                 marker_size=[40, 60, 80, 60, 40, 50],
                                 name='Month'))

        

          

###############################################################################


with modelTraning:
	import pickle


model = pickle.load(open('shop ml modell', 'rb'))



def run():
    st.header("Let's Predict")
    sel_col,disp_col=st.columns(2)
    
    
    ## For Month
    month = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    month = list(range(len(month)))
    Month = st.selectbox("Month",month, format_func=lambda x: month[x])

    
    PageValues=sel_col.slider("Enter the page values",min_value=0,max_value=300,value=10,step=30)
    ExitRates=sel_col.slider("Enter the exit rate",min_value=0.0,max_value=1.0,value=0.5,step=0.1)
    BounceRates=sel_col.slider("Enter the bounce rate",min_value=0.0,max_value=1.0,value=0.5,step=0.1)
    ProductRelated=sel_col.slider("Enter the ProductRelated",min_value=0,max_value=200,value=10,step=30)
    ProductRelated_Duration=sel_col.slider("Enter the ProductRelated Duration",min_value=0,max_value=100,value=10,step=30)
    
    if st.button("Submit"):
        
        features = [[PageValues,ExitRates,BounceRates,ProductRelated,ProductRelated_Duration,Month ]]
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = str("".join(lc))
        if ans == "True":
            st.error(
                
                'This is a Window Shopper'
            )
        else:
            st.success(
                
                'This is a potenial customer'
            )

run()
	
    

###################################################################################3

