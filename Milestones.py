import streamlit as st
import hmac
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
#import sqlalchemy
#import mysql

# Page setting
st.set_page_config(layout="wide", page_title="SkinMason Milestone Tracker")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

st.logo('SkinMason.png')
st.title('SkinMason Milestone Tracker')
#st.sidebar.title('Title')




def write_30_metrics(data):

    metrics = [{'metric': 'revenue',
                'label': 'Daily Average Revenue',
                'fmt': '${:.2f}'},
                {'metric': 'sessions',
                 'label': 'Daily Average Traffic',
                 'fmt': '{:,.0f}'},
                 {'metric': 'conversionRate',
                  'label': 'Conversion Rate',
                  'fmt':'{:.1%}'},
                  {'metric': 'averageOrderValue',
                  'label': 'Average Order Value',
                  'fmt':'${:.2f}'},
                  {'metric': 'IG_followers',
                  'label': 'Instagram Follower Count',
                  'fmt':'{:,.0f}'}]
    
    columns = st.columns(len(metrics))
    col = 0
    for metric in metrics:
        daily_value = data[metric['metric']].iloc[-1]
        delta_value = data[metric['metric']].iloc[-30]
        delta = daily_value/delta_value-1
        with columns[col]:
            st.metric(label=metric['label'],value=metric['fmt'].format(daily_value),delta='{:.0%}'.format(delta))
        col += 1

    return

def create_30day_metric(data, metric, fmt, label,help='Helpful tooltip', delta_color='normal'):
    v = data[metric].iloc[-2]
    d = data[metric].iloc[-30]
    d = v/d-1
    #label = " ".join(re.findall('[A-Z][^A-Z]*', label))
    #st.markdown(label,help='A helpful tip will go here')
    st.metric(label=label, value=fmt.format(v), delta='{:.0%}'.format(d),delta_color=delta_color,help=help)

    return

def create_gauge(data, metric,goal=655):
    value = data[metric].tail(1).values[0]
    #goal = 2.1231243*value

    percent_to_goal = round(100*value/goal,1)

    rev_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percent_to_goal,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': metric},
        number={'suffix':"%"},
        gauge={'axis':{'range': [0,100],
                       'separatethousands':True,
                       'ticksuffix':'%'},
                       'bordercolor': '#666666'}))
    
    rev_gauge.update_layout(
        margin=dict(t=10, b=10, l=35, r=35),
        width=125,
        height=105)
    rev_gauge.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    st.plotly_chart(rev_gauge)
    return

def create_follower_gauge(data, metric,goal):
    value = data[metric].tail(1).values[0]

    percent_to_goal = round(100*value/goal,1)

    rev_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percent_to_goal,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': metric},
        number={'suffix':"%"},
        gauge={'axis':{'range': [0,100],
                       'separatethousands':True,
                       'ticksuffix':'%'},
                'bordercolor': '#666666'}))
    
    rev_gauge.update_layout(
        margin=dict(t=50, b=10, l=35, r=35),
        width=125,
        height=105)
    rev_gauge.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    st.plotly_chart(rev_gauge)
    return

def create_sub_gauge(active_subs,goal):
    #value = data[metric].tail(1).values[0]

    percent_to_goal = round(100*active_subs/goal,1)

    rev_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percent_to_goal,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': 'Active Subscriptions'},
        number={'suffix':"%"},
        gauge={'axis':{'range': [0,100],
                       'separatethousands':True,
                       'ticksuffix':'%'},
                'bordercolor': '#666666'}))
    
    rev_gauge.update_layout(
        margin=dict(t=50, b=10, l=35, r=35),
        width=125,
        height=105)
    rev_gauge.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    st.plotly_chart(rev_gauge)

    return

def create_sales_indicator(data, metric, goal):
    value = data[metric].tail(1).values[0]

    percent_to_goal = round(100*value/goal,1)

    rev_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percent_to_goal,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': metric},
        number={'suffix':"%"},
        gauge={'axis':{'range': [0,100],
                       'separatethousands':True,
                       'ticksuffix':'%'},
                'shape': 'bullet',
                'bar': {'thickness': 1.0},
                'bordercolor': '#666666'}))
    
    rev_gauge.update_layout(
        margin=dict(t=0, b=00, l=100, r=35),
        width=500,
        height=50)
    rev_gauge.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    st.plotly_chart(rev_gauge)
    return

def create_sales_chart(data, metrics,colors):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for m in range(len(metrics)):
        fig1 = px.line(
            data,
            x = data.index,
            y = metrics[m]
            )
        fig1.update_traces(line=dict(color=colors[m]))
        if m != 0:
            fig1.update_traces(yaxis="y2",line=dict(color=colors[m]))
        
        fig.add_traces(fig1._data)
        fig['data'][m]['showlegend'] = True
        fig['data'][m]['name'] = metrics[m]
    #fig.update_traces(color_discrete_sequence=colors)
    fig.update_layout(
            margin=dict(t=0, b=0, l=20, r=20),
            height=250,
            showlegend=True,
            legend=dict(x=0,y=1)
        )
    fig.update_yaxes(range=[0,1.2*data[metrics[0]].max()], secondary_y=False,gridwidth=1,gridcolor='#666666')
    fig.update_xaxes(linecolor='#666666')
    fig.update_yaxes(range=[0,1.2*data[metrics[1]].max()], secondary_y=True,showgrid=False)
    fig.update_layout({
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "yaxis_tickprefix": '$',
        "yaxis_tickformat": ',.0f'})

    st.plotly_chart(fig)

    return

def create_ad_pie(data, metrics, colors=[]):
    spend_split = data[metrics].tail(30).sum() #.reset_index()
    pie_fig = px.pie(
        spend_split,
        names=spend_split.index,
        values=0,
        color_discrete_sequence=colors,
        height=250)
    
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(showlegend=False,margin=dict(t=0, b=0, l=0, r=0))
    pie_fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    st.plotly_chart(pie_fig)
    #st.write(spend_split)
    return

def create_ad_chart(data, colors):
    data['month'] = data.index.month
    plot_data = data[['month','metaAds','googleAds','revenue']].groupby('month').sum().reset_index()
    plot_data['ROAS'] = plot_data['revenue']/(plot_data['googleAds']+plot_data['metaAds'])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = px.line(
        plot_data,
        x='month',
        y='ROAS'
    )
    fig1.update_traces(yaxis="y2",line=dict(color=colors[0]))
    fig2 = px.bar(
        plot_data,
        x='month',
        y=['googleAds','metaAds'],
        color_discrete_sequence=colors[1:]
    )
    fig.add_traces(fig2._data)
    fig.add_traces(fig1._data)
    fig.update_layout({
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "legend": dict(x=0,y=1),
        "barmode": "stack",
        "margin": dict(t=0, b=0, l=20, r=20),
        "height": 250
        })
    st.plotly_chart(fig)

    return

def create_follower_chart(data, metrics, colors):
    fig = px.line(
        data,
        x = data.index,
        y=metrics,
        color_discrete_sequence=colors
        )
    #fig.update_traces(color_discrete_sequence=colors)
    fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=500,
            showlegend=True,
            legend=dict(x=0,y=1)
        )
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)","paper_bgcolor": "rgba(0, 0, 0, 0)",})
    fig.update_yaxes(gridcolor='#666666',range=[0,1.2*data[metrics[0]].max()])
    fig.update_xaxes(linecolor='#666666')
    st.plotly_chart(fig)
    return

uploaded_file = st.sidebar.file_uploader("Choose a file")


if uploaded_file is not None:

    #master_data = conn.query('SELECT * from master_data;',index_col='date')
    #rolling_data = conn.query('SELECT * from rolling_data;',index_col='date')
    #sub_counts = conn.query('SELECT * from sub_counts;',index_col='status')
    #sub_counts.drop('index',axis=1,inplace=True)
    #influencers = conn.query('SELECT * from influencers;')
    #goals = conn.query('SELECT * from milestones;',index_col='metric')
    #goals.drop('index',axis=1,inplace=True)
    conn = st.connection('gsheets', type=GSheetsConnection)
    df = conn.read()
    st.write(df)
    master_data = pd.read_excel(uploaded_file, sheet_name='master_data', index_col='date')
    rolling_data = pd.read_excel(uploaded_file, sheet_name='rolling_data', index_col='date')
    sub_counts = pd.read_excel(uploaded_file, sheet_name='sub_counts', index_col='status')
    influencers = pd.read_excel(uploaded_file, sheet_name='influencers')
    goals = pd.read_excel(uploaded_file, sheet_name='milestones', index_col='metric')

    master_data.index = pd.to_datetime(master_data.index)
    rolling_data.index = pd.to_datetime(rolling_data.index)
    influencers['deliveryDate'] = pd.to_datetime(influencers['deliveryDate'],format='%m/%d/%Y')

    cols = st.columns([.35, .3, .35])
    colors = ['#6666FF', '#FF6699', '#669966', '#FFCC00']

    #region Sales Column
    with cols[0]:
        st.markdown('## **Sales**')
        sales_metrics = {
            'revenue': {'label': 'Revenue', 'help': 'Average Daily Revenue (Last 30 Days)','fmt': '${:,.2f}'},
            'sessions': {'label': 'Web Sessions', 'help': 'Average Daily Web Sessions (Last 30 Days)', 'fmt': '{:,.0f}'},
            'conversionRate': {'label': 'Conversion Rate', 'help': 'Percent of Web Sessions Converting to final sale (Last 30 Days)','fmt': '{:.1%}'},
            'averageOrderValue': {'label': 'Average Order Value', 'help': 'Average Order Value (Last 30 Days)','fmt': '${:,.2f}'}
        }
        #sales_metrics = ['revenue', 'sessions', 'conversionRate','averageOrderValue']
        #fmts = ['${:,.2f}','{:,.0f}','{:.1%}','${:,.2f}']
        inner_cols = st.columns(len(sales_metrics))

        c=0
        for metric in sales_metrics:
            v = rolling_data[metric].iloc[-1]
            d = rolling_data[metric].iloc[-30]

            with inner_cols[c]:
                create_30day_metric(rolling_data, metric = metric, fmt=sales_metrics[metric]['fmt'], label=sales_metrics[metric]['label'], help = sales_metrics[metric]['help'])
            c+=1
        create_sales_chart(rolling_data, ['revenue','sessions'],colors=colors)

        st.markdown('Daily Average Revenue (% to Goal)',help='Measure of how close we are to reaching $20,000 monthly revenue on the last 30 days.')
        create_sales_indicator(rolling_data, 'revenue', 655)

        inner_cols_2 = st.columns(2)
        with inner_cols_2[0]:
            #st.markdown('Active Subscription Count')
            st.metric(label='Active Subscriptions',value=sub_counts.loc['ACTIVE'],help='Number of currently active subscriptions')
        with inner_cols_2[1]:
            #st.markdown('Subscription Retention Rate', help='Percent of all subscriptions ever purchased that are currently active.')
            value = sub_counts.loc['ACTIVE']/(sub_counts.loc['ACTIVE']+sub_counts.loc['CANCELLED'])
            value = value.values[0]
            st.metric(label='Subscription Retention Rate',value='{:.1%}'.format(value),help='Percent of all subscriptions ever purchased that are currently active.')
        create_sub_gauge(sub_counts.loc['ACTIVE'].values[0], 60)
    #endregion

    #region Marketing Column
    with cols[1]:
        st.markdown('## **Marketing**',help='Marketing spend data is manually updated once per week. Between manual updates, daily ad-spend is estimated based on trailing 2 weeks of actual spend')
        inner_cols = st.columns(2)
        with inner_cols[0]:
            create_30day_metric(rolling_data,'ROAS', '{:.1f}','ROAS',help='Ratio of Revenue to Ad Spend (Last 30 Days). Note that this is not strict ROAS, as it includes all sales not just those generated from ads.')
        with inner_cols[1]:
            create_30day_metric(rolling_data, 'costOfAcquisition', '${:,.2f}','Cost Of Acquisition', delta_color='inverse',help='Estimate of customer cost of acquisition, by taking total ad spend in last 30 days divided by total orders in last 30 days.')
        #create_ad_pie(master_data, ['metaAds', 'googleAds'],colors)
        create_ad_chart(master_data,colors)

        recent_influencers = influencers.sort_values('deliveryDate',ascending=False)
        recent_influencers = recent_influencers[['name', 'source', 'deliveryDate', 'status']]
        recent_influencers['deliveryDate'] = recent_influencers['deliveryDate'].dt.date
        recent_influencers = recent_influencers.head(20)

        def color_validation(val):
            if val == 'Posted':
                color = '#66CC33'
            elif val == 'En Route':
                color = '#FFCC33'
            else:
                color = '#CC3300'
            return 'background-color: %s' % color
        
        recent_influencers = recent_influencers.style.map(color_validation, subset='status')

        st.markdown('Recent Influencer Boxes')
        st.dataframe(
            recent_influencers,
            hide_index=True,
            use_container_width=True,
            column_config={
                'name': "Name",
                'source': "Source", 
                'deliveryDate': st.column_config.DateColumn(label="Date Delivered"), 
                'status': "Status"})
    #endregion

    #region Social Column
    with cols[2]:
        st.markdown('## Social',help='Historical follower counts prior to August 20, 2024 are estimated and may not be 100% accurate')
        follower_metrics = ['IG_followers', 'YT_followers','TT_followers','FB_followers']
        inner_cols = st.columns(len(follower_metrics))
        row2 = st.columns([.6,.4])

        fig = px.line()
        

        for c in range(len(follower_metrics)):
            v = rolling_data[follower_metrics[c]].iloc[-1]
            d = rolling_data[follower_metrics[c]].iloc[-30]

            with inner_cols[c]:
                create_30day_metric(master_data, follower_metrics[c], '{:,.0f}', follower_metrics[c])

            with row2[1]:
                goal = goals.loc[follower_metrics[c]].values[0]
                create_follower_gauge(rolling_data,follower_metrics[c],goal)
        
        with row2[0]:
            create_follower_chart(master_data,follower_metrics,colors)
    #endregion  






    formatted_data = master_data.copy()
    formatted_data['revenue'] = formatted_data['revenue'].map('${:,.2f}'.format)
    formatted_data['averageOrderValue'] = formatted_data['averageOrderValue'].map('${:,.2f}'.format)
    formatted_data['orders'] = formatted_data['orders'].map('{:,.0f}'.format)
    formatted_data['sessions'] = formatted_data['sessions'].map('{:,.0f}'.format)
    formatted_data['IG_followers'] = formatted_data['IG_followers'].map('{:,.0f}'.format)
    formatted_data['YT_followers'] = formatted_data['YT_followers'].map('{:,.0f}'.format)
    formatted_data['TT_followers'] = formatted_data['TT_followers'].map('{:,.0f}'.format)
    formatted_data['FB_followers'] = formatted_data['FB_followers'].map('{:,.0f}'.format)
    formatted_data['conversionRate'] = formatted_data['conversionRate'].map('{:.1%}'.format)
    with st.expander(label='Raw Data'):
        st.dataframe(formatted_data,use_container_width=True)

    with st.expander(label='Rolling 30 Day Data'):
        st.dataframe(rolling_data)



