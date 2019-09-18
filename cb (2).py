#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[126]:


df=pd.read_csv('/home/ayesha/projects/notebooks/organizations.csv')


# In[2]:


df_funding=pd.read_csv('/home/ayesha/projects/notebooks/funding_info_all.csv')


# In[5]:


df_funding[df_funding['company_name'] == 'Plum']


# In[7]:


df.head()


# In[4]:


pd.set_option('display.max_columns', None)


# In[5]:


#a=df_funding.investment_type.unique()


# In[5]:


df_funding[df_funding['company_name']=='PatientPing']


# In[7]:


df.shape


# In[142]:


#df =df.loc[df['country_code']=='USA']


# In[68]:


df.shape


# In[129]:


df =df.loc[df['status']!='closed']


# In[9]:


#df.shape[0]


# In[10]:


#df.head(5)


# In[130]:


df.reset_index(drop=True, inplace=True)


# In[8]:


df.shape


# In[131]:


df_new = df[df['category_list'].notnull() & df['founded_on'].notnull()]


# In[132]:


df_new.reset_index(drop=True, inplace=True)


# In[11]:


df_new.shape


# In[16]:


df_new.head()


# In[133]:


df_new['location']=df_new['country_code']+' '+df_new['state_code']+' '+ df_new['city']


# In[134]:


df_new['total_funding']=df_new['funding_total_usd']/(10**6)


# In[14]:


#df_new.head()


# In[135]:


import numpy as np


# In[136]:


df_new['is_ipo'] = np.where(df_new['status']=='ipo', '1', '0')


# In[137]:


df_new['is_M&A'] = np.where(df_new['status']=='acquired', '1', '0')


# In[138]:


import datetime
from dateutil import relativedelta
import datetime
now = datetime.datetime.now()


# In[139]:


now


# In[20]:


#df_new=df_new.assign(age_in_months='')


# In[21]:


df.head()


# In[140]:


df_new['founded_on']=df_new['founded_on'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d') )


# In[141]:


#df_new.head()


# In[142]:


df_new['age_in_months']=df_new['founded_on'].apply(lambda x:12*relativedelta.relativedelta(now,x ).years+relativedelta.relativedelta(now,x ).months )


# In[143]:


df_new['age']=df_new["age_in_months"]/12


# In[144]:


df_new.drop(columns = 'age_in_months' , inplace = True)


# In[8]:


#df_new.head()


# In[145]:


df_AIML = df_new.copy(deep=True)


# In[146]:


df_fintech=df_new.copy(deep=True)


# In[147]:


df_healthcare = df_new.copy(deep=True)


# In[148]:


df_healthcare = pd.DataFrame().reindex_like(df_new)


# In[149]:


df_AIML = pd.DataFrame().reindex_like(df_new)


# In[150]:


df_fintech = pd.DataFrame().reindex_like(df_new)


# In[151]:


df_healthcare.shape


# In[152]:


healthcare=['Medical','First Aid','Assistive Technology','Cosmetic Surgery','Fertility','Medical Device','Nursing and Residential Care','Clinical Trials','Alternative Medicine','Elderly','Dental','mHealth','Home Health Care','Personal Health','Assisted Living','Emergency Medicine','Hospital','Health Diagnostics','Electronic Health Record (EHR)','Outpatient Care','Elder Care','Wellness','Diabetes','Baby','Therapeutics']
AIML = ['Artificial Intelligence','E-Learning','Data Visualization','Facial Recognition','Prediction Markets','Spam Filtering','Speech Recognition','Marketing Automation','Text Analytics','Smart Home','Natural Language Processing','Augmented Reality','Predictive Analytics','Big Data','Data Mining','Autonomous Vehicles','Image Recognition','Fraud Detection','Analytics','Robotics','Semantic Search','Intelligent Systems','Business Intelligence','Machine Learning','Computer Vision','Smart Building']


# In[153]:


fintech=['Payments','Prediction Markets','Blockchain','Personal Finance','Financial Exchange','Finance','Financial Services','Mobile Payments','FinTech','Insurance','Banking']


# In[154]:


cat_string = "|".join(healthcare)
df_new['category_list'] = df_new['category_list'].fillna('')
df_healthcare= df_new[df_new['category_list'].str.contains(cat_string)]


# In[155]:


df_healthcare.shape


# In[156]:


cat_string = "|".join(AIML)
df_new['category_list'] = df_new['category_list'].fillna('')
df_AIML= df_new[df_new['category_list'].str.contains(cat_string)]


# In[157]:


df_AIML.shape


# In[158]:


cat_string = "|".join(fintech)
df_new['category_list'] = df_new['category_list'].fillna('')
df_fintech= df_new[df_new['category_list'].str.contains(cat_string)]


# In[159]:


df_fintech.shape


# In[160]:


#df_healthcare.head()


# In[45]:


#df_AIML.head()


# In[186]:


df_funding.head()


# In[161]:


df_AIML['investment_type']=''
df_healthcare['investment_type']=''
df_fintech['investment_type']=''


# In[162]:


df_healthcare.reset_index(drop=True, inplace=True)
df_AIML.reset_index(drop=True, inplace=True)
df_fintech.reset_index(drop=True, inplace=True)


# In[163]:


import warnings
warnings.filterwarnings('ignore')


# In[192]:


#df_funding.sort_values(by='announced_on',ascending=False)


# In[164]:


df_healthcare['founded_on']=pd.to_datetime(df_healthcare['founded_on'],errors='coerce')


# In[165]:


df_healthcare['last_funding_on']=pd.to_datetime(df_healthcare['last_funding_on'],errors='coerce')


# In[166]:


df_healthcare['LFY']=df_healthcare['last_funding_on'].subtract(df_healthcare['founded_on'])


# In[167]:


df_healthcare['LFY']=df_healthcare["LFY"].dt.days/365


# In[53]:


df_healthcare.head()


# In[168]:


df_AIML['founded_on']=pd.to_datetime(df_AIML['founded_on'],errors='coerce')
df_AIML['last_funding_on']=pd.to_datetime(df_AIML['last_funding_on'],errors='coerce')
df_AIML['LFY']=df_AIML['last_funding_on'].subtract(df_AIML['founded_on'])


# In[169]:


df_AIML['LFY']=df_AIML["LFY"].dt.days/365


# In[170]:


df_fintech['founded_on']=pd.to_datetime(df_fintech['founded_on'],errors='coerce')
df_fintech['last_funding_on']=pd.to_datetime(df_fintech['last_funding_on'],errors='coerce')
df_fintech['LFY']=df_fintech['last_funding_on'].subtract(df_fintech['founded_on'])


# In[171]:


df_fintech['LFY']=df_fintech["LFY"].dt.days/365


# In[172]:


#df_fintech.head()


# In[59]:


df_funding.drop_duplicates(subset=['company_name'],keep='first',inplace=True)


# In[111]:


df_funding[df_funding['company_name']=='Nivo']


# In[325]:


df_funding.shape


# In[173]:


dict_investment = dict(zip(df_funding['company_name'], df_funding['investment_type']))


# In[174]:


#dict_investment


# In[175]:


df_AIML['investment_type']=df_AIML['company_name'].map(dict_investment)


# In[176]:


df_healthcare['investment_type']=df_healthcare['company_name'].map(dict_investment)


# In[177]:


df_fintech['investment_type']=df_fintech['company_name'].map(dict_investment)


# In[178]:


df_AIML['investment_type'].unique()


# In[213]:


df_healthcare['investment_type'].value_counts()


# In[214]:


df_healthcare['investment_type'].value_counts()


# In[215]:


df_fintech['investment_type'].unique()


# In[179]:


df_healthcare['industry_vertical']='healthcare'


# In[180]:


df_AIML['industry_vertical']='AIML'


# In[181]:


df_fintech['industry_vertical']='Fintech'


# In[182]:


df_healthcare['location']=df_healthcare['country_code']+' '+df_healthcare['state_code']+' '+ df_healthcare['city']


# In[183]:


df_AIML['location']=df_AIML['country_code']+' '+df_AIML['state_code']+' '+ df_AIML['city']


# In[184]:


df_fintech['location']=df_fintech['country_code']+' '+df_fintech['state_code']+' '+ df_fintech['city']


# In[70]:


df_healthcare.head()


# In[185]:


df_healthcare = df_healthcare[['company_name','total_funding','is_ipo','is_M&A','age','LFY','location','investment_type','industry_vertical']]


# In[186]:


df_AIML = df_AIML[['company_name','total_funding','is_ipo','is_M&A','age','LFY','location','investment_type','industry_vertical']]


# In[187]:


df_fintech = df_fintech[['company_name','total_funding','is_ipo','is_M&A','age','LFY','location','investment_type','industry_vertical']]


# In[74]:


df_AIML.shape


# In[188]:


import datetime
from dateutil import relativedelta
import datetime
now = datetime.datetime.now()


# In[134]:


#df_healthcare['age']=df_healthcare["age_in_months"].dt.days/365


# In[ ]:





# In[ ]:





# In[345]:


#df_healthcare['Age'] = ['1' if x<=12 else '2' if 12<x<=24 else '3' if 24<x<=36 else '4' if 36<x<=48 else '5' if 48<x<=60 else '6-10' if 60<x<=120 else '>10' for x in df_healthcare['age_in_months']]


# In[346]:


#df_AIML['Age'] = ['1' if x<=12 else '2' if 12<x<=24 else '3' if 24<x<=36 else '4' if 36<x<=48 else '5' if 48<x<=60 else '6-10' if 60<x<=120 else '>10' for x in df_AIML['age_in_months']]


# In[347]:


#df_fintech['Age'] = ['1' if x<=12 else '2' if 12<x<=24 else '3' if 24<x<=36 else '4' if 36<x<=48 else '5' if 48<x<=60 else '6-10' if 60<x<=120 else '>10' for x in df_fintech['age_in_months']]


# In[76]:


df_healthcare.head()


# In[189]:


df_healthcare['investment_type']=df_healthcare['investment_type'].fillna('nan')


# In[190]:


df_fintech['investment_type']=df_fintech['investment_type'].fillna('nan')


# In[191]:


df_AIML['investment_type']=df_AIML['investment_type'].fillna('nan')


# In[192]:


df_healthcare['is_ipo']=df_healthcare['is_ipo'].apply(lambda x: bool(int(x)))


# In[193]:


df_AIML['is_ipo']=df_AIML['is_ipo'].apply(lambda x: bool(int(x)))


# In[194]:


df_fintech['is_ipo']=df_fintech['is_ipo'].apply(lambda x: bool(int(x)))


# In[195]:


df_healthcare.head()


# In[196]:


df_healthcare['is_M&A']=df_healthcare['is_M&A'].apply(lambda x: bool(int(x)))


# In[197]:


df_AIML['is_M&A']=df_AIML['is_M&A'].apply(lambda x: bool(int(x)))


# In[198]:


df_fintech['is_M&A']=df_fintech['is_M&A'].apply(lambda x: bool(int(x)))


# In[199]:


df_fintech.head()


# In[200]:


df_healthcare['Exit']=df_healthcare['is_ipo'] | df_healthcare['is_M&A']


# In[201]:


df_AIML['Exit']=df_AIML['is_ipo'] | df_AIML['is_M&A']


# In[202]:


df_fintech['Exit']=df_fintech['is_ipo'] | df_fintech['is_M&A']


# In[243]:


df_healthcare.head()


# In[203]:


df_com = pd.concat([df_healthcare, df_AIML,df_fintech])


# In[ ]:





# In[245]:


df_com['investment_type'].value_counts()


# In[204]:


df_com.loc[df_com['investment_type']=='nan','total_funding']=0


# In[205]:


df_com[df_com['total_funding']==0]


# In[206]:


df_com['total_funding']=df_com['total_funding'].fillna('undisclosed')


# In[369]:


df_com[df_com['total_funding']=='undisclosed']


# In[207]:


df_com['funding_total'] = ['undisclosed' if x=='undisclosed' else '0' if x==0 else '<1' if x <=1 else '1-10' if 1<x<=10 else '10-50' if 10<x<=50  else '50-100' if 50<x<=100 else '100+' for x in df_com['total_funding']]


# In[93]:


df_com.head()


# In[208]:


orgs=pd.read_csv('/home/ayesha/projects/notebooks/organizations.csv')


# In[209]:


orgs['employee_count'].unique()


# In[210]:


dict_emp = dict(zip(orgs['company_name'], orgs['employee_count']))


# In[98]:


dict_emp


# In[211]:


dict_state = dict(zip(orgs['company_name'], orgs['state_code']))


# In[212]:


df_com['state']=df_com['company_name'].map(dict_state)


# In[213]:


df_com['no_of_employees']=df_com['company_name'].map(dict_emp)


# In[102]:


df_com.head()


# In[214]:


dict_city = dict(zip(orgs['company_name'], orgs['city']))


# In[215]:


df_com['city']=df_com['company_name'].map(dict_city)


# In[138]:


df_com.head()


# In[ ]:


dict1 = dict(zip(date_and_fund['company_name']))


# In[ ]:





# In[ ]:





# In[216]:


df_com.drop(columns=['is_ipo', 'is_M&A','location'],inplace=True)


# In[137]:


df_com.shape


# In[217]:


dict_country = dict(zip(orgs['company_name'], orgs['country_code']))


# In[218]:


df_com['Country']=df_com['company_name'].map(dict_country)


# In[16]:


df_com = pd.read_excel('/home/ayesha/projects/notebooks/sevnteensept.xlsx')


# In[6]:


fund = pd.read_csv('/home/ayesha/projects/notebooks/funding_info_all.csv')


# In[7]:


fund.sort_values(by = ["announced_on"],ascending=True,inplace=True)


# In[11]:


fund['investment_type'].unique()


# In[12]:


invest = ['seed', 'undisclosed', 'series_unknown', 'corporate_round',
       'private_equity', 'series_a', 'grant', 'series_b', 'pre_seed',
       'series_c', 'angel', 'series_d', 'debt_financing',
       'post_ipo_equity', 'convertible_note', 'series_f', 'series_e',
       'secondary_market', 'series_g', 'series_h',
       'non_equity_assistance', 'post_ipo_debt', 'equity_crowdfunding',
       'product_crowdfunding', 'post_ipo_secondary',
       'initial_coin_offering', 'series_i', 'series_j']


# In[13]:


fund=fund[fund['investment_type'].isin(invest)]


# In[17]:


fund=fund[fund['company_name'].isin(df_com[df_com['industry_vertical']=='healthcare']['company_name'])]


# In[65]:


date_and_fund1=fund[~fund.duplicated(subset=['company_name','investment_type'],keep='first')].groupby('company_name').apply(lambda x: dict(zip(x['announced_on'],x['investment_type']))).reset_index().rename(columns={0:'date_and_funding'})


# In[66]:


date_and_fund2.drop(columns= 'company_name' , inplace = True)


# In[57]:


date_and_fund2=fund[~fund.duplicated(subset=['company_name','investment_type'],keep='first')].groupby('company_name').apply(lambda x: dict(zip(x['announced_on'], x['raised_amount_usd'].astype(str)))).reset_index().rename(columns={0:'date_and_funds'})


# In[68]:


date_and_fund.head()


# In[67]:


date_and_fund = pd.concat([date_and_fund1, date_and_fund2], axis=1)


# In[23]:


pd.options.display.max_colwidth = 500


# In[69]:


#fund[fund['company_name'] == '10X Genomics']


# In[71]:


date_and_fund['date'] = date_and_fund['date_and_funding'].apply(lambda x : x.keys())


# In[73]:


date_and_fund['funding'] = date_and_fund['date_and_funding'].apply(lambda x : x.values())


# In[75]:


date_and_fund['amount'] = date_and_fund['date_and_funds'].apply(lambda x : x.values())


# In[32]:


#new = date_and_fund["funding"].str.split(" ", n = 1, expand = True)


# In[77]:


date_and_fund.drop(columns=['date_and_funding','date_and_funds'] , inplace =True)


# In[90]:


b = date_and_fund.copy(deep = True)


# In[224]:


di = dict(zip(fund['company_name'] , fund['state_code_y']))


# In[110]:


df1 = pd.DataFrame(date_and_fund.funding.values.tolist(), index=date_and_fund.company_name)


# In[112]:


df2 = pd.DataFrame(date_and_fund.date.values.tolist(), index=date_and_fund.company_name)


# In[113]:


df3 = pd.DataFrame(date_and_fund.amount.values.tolist(), index=date_and_fund.company_name)


# In[120]:


b1=df2.stack().reset_index(drop=True, level=1).reset_index(name='date')


# In[121]:


c1=df3.stack().reset_index(drop=True, level=1).reset_index(name='amount')


# In[126]:


c1.head()


# In[118]:


a1=df1.stack().reset_index(drop=True, level=1).reset_index(name='funding')


# In[125]:


c1.drop(columns= 'company_name' , inplace = True)


# In[128]:


date_and_fund = pd.concat([a1,b1,c1] , axis = 1)


# In[134]:


c = pd.DataFrame(date_and_fund.groupby('company_name'))


# In[136]:


c.head()


# In[225]:


df_com['Investor_State'] = df_com['company_name'].map(di)


# In[139]:


date_and_fund.head()


# In[142]:


#dict1 = dict(zip(date_and_fund['company_name'] , date_and_fund['date']))


# In[143]:


#dict1


# In[ ]:


date_and_fund[]


# In[ ]:





# In[226]:


df_com.head()


# In[227]:


df_com.to_excel('/home/ayesha/projects/notebooks/sevnteensept.xlsx' , index = False)


# In[228]:


df_com[(df_com['industry_vertical'] == 'healthcare') & (df_com['Country'] == 'USA')].shape


# In[265]:


a =df_com[(df_com['industry_vertical'] == 'healthcare') & (df_com['Country'] == 'USA')]


# In[266]:


a.describe()


# In[267]:


#b = a.sort_values('age' , ascending = False)


# In[268]:


a['Investor_State'].value_counts()


# In[269]:


b = a[['company_name' , 'age' , 'state' , 'Country' , 'Investor_Name' , 'Investor_State' , 'Investor_Country']]


# In[256]:


b = b.sort_values('age')


# In[270]:


b.to_excel('/home/ayesha/projects/notebooks/regional.xlsx',index = False)


# In[271]:


b.head()


# In[108]:


df_com.shape


# In[265]:


mylist = ["B2B", "B2C"]
import random


# In[266]:


df_com['type_of_startup']=random.choices(mylist, weights = [30, 70], k = 48971)


# In[267]:


founders=['yes','no']


# In[268]:


success = ['0%' , '0-5%' , '5-10%', '10+%']


# In[269]:


revenue = ['commission-based' , 'fee-for-service' , 'Advertising', 'SaaS' ,'Other']


# In[270]:


df_com['founders_prev_startup_exp']=random.choices(founders, weights = [5, 95], k = 48971)


# In[271]:


df_com['founders_prev_fundraising_exp']=random.choices(founders, weights = [5, 95], k = 48971)


# In[272]:


df_com['previous_exit_of_founders'] = random.choices(founders , weights = [15 , 85] , k = 48971)


# In[273]:


df_com['previous_exits_by_lead_investors'] = random.choices(founders , weights = [10 , 90] , k = 48971)


# In[274]:


df_com['success_rate_of_lead_investors'] = random.choices(success , weights = [72 , 14, 9, 5] , k = 48971)


# In[275]:


df_com['revenue_model'] = random.choices(revenue , weights = [5 , 5, 10, 70, 10] , k = 48971)


# # ADD KNOW SUPPORT COL

# In[263]:


# investor type latest or what ask sir
#dict_investor = dict(zip(df_funding['company_name'],df_funding['investor_type']))


# In[264]:


dict_investor


# In[236]:


df_com['knowledge_support'] = df_com['company_name'].map(dict_investor)


# In[237]:


req_investor = ['accelerator','incubator','university_program']


# In[238]:


df_com['knowledge_support_bool'] = df_com['knowledge_support'].apply(lambda x : 'yes' if x in req_investor else 'no')


# In[239]:


df_com.drop(columns = ['knowledge_support'] , inplace= True)


# In[240]:


df_com.rename(columns = {'knowledge_support_bool' : 'knowledge_support'} , inplace = True)


# In[400]:


df_com.shape


# In[277]:


df_com['Exit']=df_com['Exit'].apply(lambda x: 'yes' if x else 'no')


# In[278]:


df_com.head()


# In[6]:


df_com.to_excel('/home/ayesha/projects/notebooks/final_fintech_healthcare_aiml_companies.xlsx',index=False)


# In[ ]:


df_com


# # GRAPHS SNIPPET

# In[7]:


df_h=df_com[df_com['industry_vertical']=='healthcare']


# In[269]:


df_com.shape


# In[270]:


df_com.head(100)


# In[2]:


import seaborn as sns
from matplotlib.pyplot import show
import matplotlib.pyplot as plt


# ## Healthcare Graphs

# In[141]:



sns.set(style="darkgrid")
ax = sns.countplot(x='Age',order=["1", "2",'3','4','5','6-10','>10'],data=df_h)
total = float(len(df_h))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
show()


# In[142]:


ax.figure.savefig("/home/Ayesha/notebooks/healthcare/age.png")


# In[143]:


sns.set(style="darkgrid")
plt.figure(figsize=(20,5))
ax1 = sns.countplot(x='no_of_employees',order=["1-10", "11-50",'51-100','101-250','251-500','501-1000','1001-5000','5001-10000','10000+','unknown'],data=df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[144]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/no_of_employee.png")


# In[324]:


df_com['state'].unique()


# In[11]:


sns.set(style="darkgrid")
#sns.set_context("poster", font_scale=5)
plt.figure(figsize=(80,20))
ax1 = sns.countplot(x='investment_type',order=['seed', 'private_equity', 'series_unknown', 'series_a', 'nan',
       'undisclosed', 'debt_financing', 'grant', 'convertible_note',
       'series_b', 'angel', 'post_ipo_debt', 'equity_crowdfunding',
       'post_ipo_equity', 'product_crowdfunding', 'non_equity_assistance',
       'series_d', 'series_c', 'pre_seed', 'corporate_round',
       'secondary_market', 'initial_coin_offering', 'series_f',
       'series_e'],data=df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[12]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/last_funding_status.png")


# In[287]:


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=3.8)
plt.figure(figsize=(80,10))
ax1 = sns.countplot(x='state',order=['IL', 'TX', 'NY', 'FL', 'GA', 'PA', 'CA', 'DC', 'MN', 'MA', 'AZ',
       'NH', 'TN', 'WA', 'NC', 'nan', 'CO', 'NJ', 'BC', 'MD', 'DE', 'WI',
       'VA', 'CT', 'VT', 'OH', 'NV', 'MI', 'MO', 'KS', 'AL', 'IN', 'ID',
       'KY', 'AR', 'RI', 'UT', 'HI', 'OR', 'AB', 'SC', 'IA', 'WY', 'LA',
       'ME', 'NM', 'WV', 'MT', 'OK', 'NE', 'MS', 'SD', 'AK', 'ON', 'ND',
       'QC', 'MB', 'NB', 'NS'],data=df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[288]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/location_state.png")


# In[249]:


df_h['total_funding']=df_h['total_funding'].fillna(0)
df_h.head(100)


# In[250]:


df_h['funding_total'].unique()


# In[268]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['funding_total']
ax1 = sns.countplot(x , order = ['0' , '<1','1-10','10-50','50-100','100+'] , data=df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[269]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/total_funding.png")


# In[270]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['Exit']
ax1 = sns.countplot(x , order = ['yes' , 'no'])
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[271]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/exit.png")


# In[272]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['type_of_startup']
ax1 = sns.countplot(x , order = ['B2B','B2C'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[275]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/type_startup.png")


# In[319]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['founders_prev_startup_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[321]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/founders_prev_startup_exp.png")


# In[277]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['founders_prev_fundraising_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[278]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/founders_prev_fundraising_exp.png")


# In[322]:


df_com.head()


# In[323]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['knowledge_support']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[324]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/knowledge_support.png")


# In[327]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_h['revenue_model']
ax1 = sns.countplot(x , order = ['SaaS', 'Other', 'Advertising', 'commission-based',
       'fee-for-service'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[328]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/revenue_model.png")


# In[330]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_h['success_rate_of_lead_investors']
ax1 = sns.countplot(x , order = ['0%','0-5%','5-10%',  '10+%' ] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[331]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/success_rate_of_lead_investors.png")


# In[332]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['previous_exits_by_lead_investors']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[333]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/previous_exits_by_lead_investors.png")


# In[334]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_h['previous_exit_of_founders']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_h)
total = float(len(df_h))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()#previous_exit_of_founders


# In[335]:


ax1.figure.savefig("/home/Ayesha/notebooks/healthcare/previous_exit_of_founders.png")


# In[336]:


df_h.shape


# In[ ]:





# In[13]:


df_f=df_com[df_com['industry_vertical']=='Fintech']


# In[14]:


df_f.shape


# ## Fintech Graphs

# In[339]:


import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
ax = sns.countplot(x='Age',order=["1", "2",'3','4','5','6-10','>10'],data=df_f)
total = float(len(df_f))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
show()


# In[340]:


ax.figure.savefig("/home/Ayesha/notebooks/fintech/age.png")


# In[341]:


sns.set(style="darkgrid")
plt.figure(figsize=(20,5))
ax1 = sns.countplot(x='no_of_employees',order=["1-10", "11-50",'51-100','101-250','251-500','501-1000','1001-5000','5001-10000','10000+','unknown'],data=df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[342]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/no_of_employees.png")


# In[324]:


df_com['state'].unique()


# In[15]:


sns.set(style="darkgrid")
#sns.set_context("notebook", font_scale=3.8)
plt.figure(figsize=(80,10))
ax1 = sns.countplot(x='investment_type',order=['seed', 'private_equity', 'series_unknown', 'series_a', 'nan',
       'undisclosed', 'debt_financing', 'grant', 'convertible_note',
       'series_b', 'angel', 'post_ipo_debt', 'equity_crowdfunding',
       'post_ipo_equity', 'product_crowdfunding', 'non_equity_assistance',
       'series_d', 'series_c', 'pre_seed', 'corporate_round',
       'secondary_market', 'initial_coin_offering', 'series_f',
       'series_e'],data=df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[16]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/last_funding_status.png")


# In[344]:


sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=3.8)
plt.figure(figsize=(80,10))
ax1 = sns.countplot(x='state',order=['IL', 'TX', 'NY', 'FL', 'GA', 'PA', 'CA', 'DC', 'MN', 'MA', 'AZ',
       'NH', 'TN', 'WA', 'NC', 'nan', 'CO', 'NJ', 'BC', 'MD', 'DE', 'WI',
       'VA', 'CT', 'VT', 'OH', 'NV', 'MI', 'MO', 'KS', 'AL', 'IN', 'ID',
       'KY', 'AR', 'RI', 'UT', 'HI', 'OR', 'AB', 'SC', 'IA', 'WY', 'LA',
       'ME', 'NM', 'WV', 'MT', 'OK', 'NE', 'MS', 'SD', 'AK', 'ON', 'ND',
       'QC', 'MB', 'NB', 'NS'],data=df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[345]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/location_state.png")


# In[346]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['funding_total']
ax1 = sns.countplot(x , order = ['0' , '<1','1-10','10-50','50-100','100+'],data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[347]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/total_funding.png")


# In[348]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['type_of_startup']
ax1 = sns.countplot(x , order = ['B2B','B2C'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[349]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/type_startup.png")


# In[350]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['founders_prev_startup_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[351]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/founders_prev_startup_exp.png")


# In[352]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['founders_prev_fundraising_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[353]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/founders_prev_fundraising_exp.png")


# In[354]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['Exit']
ax1 = sns.countplot(x , order = ['yes' , 'no'])
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[355]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/exit.png")


# In[356]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['knowledge_support']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[357]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/knowledge_support.png")


# In[358]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_f['revenue_model']
ax1 = sns.countplot(x , order = ['SaaS', 'Other', 'Advertising', 'commission-based',
       'fee-for-service'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[359]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/revenue_model.png")


# In[360]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_f['success_rate_of_lead_investors']
ax1 = sns.countplot(x , order = ['0%','0-5%','5-10%',  '10+%' ] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[361]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/success_rate_of_lead_investors.png")


# In[362]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['previous_exits_by_lead_investors']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[363]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/previous_exits_by_lead_investors.png")


# In[364]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_f['previous_exit_of_founders']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_f)
total = float(len(df_f))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()#previous_exit_of_founders


# In[365]:


ax1.figure.savefig("/home/Ayesha/notebooks/fintech/previous_exit_of_founders.png")


# In[ ]:





# In[17]:


df_a=df_com[df_com['industry_vertical']=='AIML']


# In[18]:


df_a.shape


# In[368]:


df_a.head(100)


# ## AIML Graphs

# In[369]:


import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
ax = sns.countplot(x='Age',order=["1", "2",'3','4','5','6-10','>10'],data=df_a)
total = float(len(df_a))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
show()


# In[370]:


ax.figure.savefig("/home/Ayesha/notebooks/AIML/age.png")


# In[371]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(20,5))
ax1 = sns.countplot(x='no_of_employees',order=["1-10", "11-50",'51-100','101-250','251-500','501-1000','1001-5000','5001-10000','10000+','unknown'],data=df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
show()


# In[372]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/no_of_employees.png")


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
#sns.set_context("notebook" , font_scale= 3)
plt.figure(figsize=(80,10))
ax1 = sns.countplot(x='investment_type',order=['seed', 'private_equity', 'series_unknown', 'series_a', 'nan',
       'undisclosed', 'debt_financing', 'grant', 'convertible_note',
       'series_b', 'angel', 'post_ipo_debt', 'equity_crowdfunding',
       'post_ipo_equity', 'product_crowdfunding', 'non_equity_assistance',
       'series_d', 'series_c', 'pre_seed', 'corporate_round',
       'secondary_market', 'initial_coin_offering', 'series_f',
       'series_e'],data=df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
show()


# In[21]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/last_funding_status.png")


# In[373]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set_context("notebook" , font_scale= 3.8)
plt.figure(figsize=(80,10))
ax1 = sns.countplot(x='state',order=['IL', 'TX', 'NY', 'FL', 'GA', 'PA', 'CA', 'DC', 'MN', 'MA', 'AZ',
       'NH', 'TN', 'WA', 'NC', 'nan', 'CO', 'NJ', 'BC', 'MD', 'DE', 'WI',
       'VA', 'CT', 'VT', 'OH', 'NV', 'MI', 'MO', 'KS', 'AL', 'IN', 'ID',
       'KY', 'AR', 'RI', 'UT', 'HI', 'OR', 'AB', 'SC', 'IA', 'WY', 'LA',
       'ME', 'NM', 'WV', 'MT', 'OK', 'NE', 'MS', 'SD', 'AK', 'ON', 'ND',
       'QC', 'MB', 'NB', 'NS'],data=df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[374]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/location_state.png")


# In[375]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['funding_total']
ax1 = sns.countplot(x , order = ['0' , '<1','1-10','10-50','50-100','100+'],data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[376]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/total_funding.png")


# In[377]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['type_of_startup']
ax1 = sns.countplot(x , order = ['B2B','B2C'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[378]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/type_startup.png")


# In[379]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['founders_prev_startup_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[380]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/founders_prev_startup_exp.png")


# In[387]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['founders_prev_fundraising_exp']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[388]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/founders_prev_fundraising_exp.png")


# In[383]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['Exit']
ax1 = sns.countplot(x , order = ['yes' , 'no'])
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[384]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/exit.png")


# In[385]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['knowledge_support']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[386]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/knowledge_support.png")


# In[389]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_a['revenue_model']
ax1 = sns.countplot(x , order = ['SaaS', 'Other', 'Advertising', 'commission-based',
       'fee-for-service'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[390]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/revenue_model.png")


# In[391]:


sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
x = df_a['success_rate_of_lead_investors']
ax1 = sns.countplot(x , order = ['0%','0-5%','5-10%',  '10+%' ] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[392]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/success_rate_of_lead_investors.png")


# In[393]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['previous_exits_by_lead_investors']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[394]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/previous_exits_by_lead_investors.png")


# In[395]:


sns.set(style="darkgrid")
plt.figure(figsize=(7,5))
x = df_a['previous_exit_of_founders']
ax1 = sns.countplot(x , order = ['yes','no'] ,data = df_a)
total = float(len(df_a))

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height+10,
            '{0:.0%}'.format(height/total),
            ha="center") 
plt.show()


# In[396]:


ax1.figure.savefig("/home/Ayesha/notebooks/AIML/previous_exit_of_founders.png")


# In[405]:


df_com.head()


# In[406]:


df_com.drop(columns = ['total_funding','city','Country'] , inplace = True)


# In[407]:


df_com.shape


# In[408]:


import pandas as pd


# In[5]:


#df1 = pd.read_excel("final_fintech_healthcare_aiml_companies.xlsx")


# In[6]:


#df_com = df1.copy(deep = True)


# In[412]:


df_com.head()


# In[410]:


df_com.rename(columns = {'state' : 'location' } , inplace = True)


# In[411]:


df_com.rename(columns = {'investment_type' : 'last_funding_status' } , inplace = True)


# In[419]:


def cohort(**kwargs) :
    filters=True
    for key, value in kwargs.items(): 
        filters=filters& (df_com[key]==value)
    df_coho=df_com[filters]
    data = []
    
    # Total No of Companies
    data.append(df_coho.shape[0])
    
    # No of Funded Companies
    try :
        data.append(df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])
    except:
        data.append(df_coho.shape[0])
    
    # No of Funded Companies Likelihood
    try :
        data.append((df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])*100/df_coho.shape[0])
    except:
        data.append(100)
        
    # No of Non Funded Companies
    try:
        data.append(df_coho['last_funding_status'].value_counts()['nan'])
    except:
        data.append(0)
        
    # No of Non Funded Companies Likelihood
    try:
        data.append((df_coho['last_funding_status'].value_counts()['nan']*100/df_coho.shape[0]))
    except:
        data.append(0)
   
    # No of Company Exited
    try:
        data.append(df_coho['Exit'].value_counts()['yes'])
    except:
        data.append(0)
        
     # No of Company Exited Likelihood    
    try:
        data.append((df_coho['Exit'].value_counts()['yes']*100/df_coho.shape[0]))
    except:
        data.append(0)
        
    # List to store index of 'last_funding_status' which are not null 
    last_funding= [i for i in df_coho['last_funding_status'].unique() if i!='nan'] 
    # Series to store count of all index
    ser=df_coho['last_funding_status'].value_counts()
    
    
    # List to store count of not nulll indices  
    last_funding_count=[]
    for i in last_funding:
        last_funding_count.append(ser[i])
    
    # Likelihood of all indices other than 'nan'
    funding_likelihood=[(i*100)/df_coho.shape[0] for i in last_funding_count]
    
    
    # Initialization of workbook
    workbook = xlsxwriter.Workbook('/home/ayesha/projects/notebooks/healthcare.xlsx') 
    worksheet = workbook.add_worksheet() 
  

    row = 0
    column = 0

    content = ["Total No Of Companies", "Funded Companies Count", "Funded Companies Likelihood", 
               "Non Funded Companies Count", 
                        "Non Funded Companies Likelihood", 
                 "Exit Count","Exit Likelihood","Funded Companies Distribution",
               "Funded Companies Distribution Count","Funding Likelihood", 
               ] 

    # Headers  
    for item in content : 

        # write operation perform 
        worksheet.write(row, column, item) 

        # incrementing the value of row by one 
        # with each iteratons. 
        column += 1
    
    # Single row ,Column values 
    column = 0
    for item in data : 
        # write operation perform 
        worksheet.write(row +1, column, item) 

        # incrementing the value of row by one 
        # with each iteratons. 
        column += 1
    
    # Multiple rows of 'last_funding_status' excluding 'nan'
    row=1    
    for item in last_funding:
        worksheet.write(row,column,item)
        row+=1
    
    # Multiple rows of 'last_funding_status' count excluding 'nan'
    row=1
    column+=1
    for item in last_funding_count:
        worksheet.write(row,column,item)
        row+=1
    
    # Multiple rows of 'last_funding_status' likelihood excluding 'nan'
    row=1
    column+=1
    for item in funding_likelihood:
        worksheet.write(row,column,item)
        row+=1
    
    
    workbook.close() 
    
    

    return df_coho.head()
    


# In[420]:


cohort(Age='4',industry_vertical='healthcare',no_of_employees='11-50',location='MA')


# In[421]:


df_com['last_funding_status'].unique()


# In[425]:


df_com[(df_com['industry_vertical']=='AIML')]


# In[447]:


df_com[(df_com['Exit']=='no')&(df_com['last_funding_status']!='nan')&(df_com['industry_vertical']=='AIML')]['last_funding_status'].value_counts()


# In[448]:


df_com['Age'].unique()


# In[760]:


def state_age(**kwargs) :
    filters=True
    for key, value in kwargs.items(): 
        filters=filters& (df_com[key]==value)
    df_coho=df_com[filters]
    data = []
    
    # Total No of Companies
    data.append(df_coho.shape[0])
    
    # Total No of Funded Companies
    try :
        data.append(df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])
    except:
        data.append(df_coho.shape[0])
        
    # Likelihood of Funded Companies 
    try :
        data.append((df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])*100/df_coho.shape[0])
    except:
        data.append(100)

    # No of exited companies
    try:
        data.append(df_coho['Exit'].value_counts()['yes']-
                    df_coho[df_coho['Exit']=='yes']['last_funding_status'].value_counts()['nan'])
    except:
        data.append(0)
        
    #ask sir
    # Likelihood of exited companies
    try:
        data.append(((df_coho['Exit'].value_counts()['yes'] - 
                      df_coho[df_coho['Exit']=='yes']['last_funding_status'].value_counts()['nan'])*100/df_coho.shape[0]))
    except:
        data.append(0)
    
    # List of 'last_funding_status' except 'nan'  
    last_funding= [i for i in df_coho['last_funding_status'].unique() if i!='nan'] 
    # Required indices of 'las_funding_status'
    check_funding=['seed', 'private_equity', 'series_a',
        'debt_financing', 'grant', 
       'series_b', 'angel','series_d', 'series_c', 'pre_seed', 
         'series_f','series_e']
    # Intersection of 'last_funding' and 'check_funding'
    final_funding_types=set(last_funding).intersection(check_funding) 
    
    # Series of count of 'last_funding_status' indices 
    ser=df_coho[df_coho['Exit']=='no']['last_funding_status'].value_counts()
    
    # List to store count of 'last_funding_status' indices 
    last_funding_count=[]
    for i in final_funding_types:
        last_funding_count.append(ser[i])
    
    # Funding likelihood of 'last_funding_status' indices
    funding_likelihood=[(i*100)/df_coho.shape[0] for i in last_funding_count]
    
    # Age
    age_exit_ser=df_coho[(df_coho['Exit']=='yes')&(df_coho['last_funding_status']!='nan')]['Age'].value_counts()
    age_exit=list(df_coho[(df_coho['Exit']=='yes')&(df_coho['last_funding_status']!='nan')]['Age'].unique())
    
    for i in range(0,len(age_exit)):
        if age_exit[i]=='6-10':
            age_exit[i]='8'
        elif age_exit[i]=='>10':
            age_exit[i]='15'
        else:
            continue
    age_exit = list(map(int, age_exit))
    min_age_exit=min(age_exit)
    if min_age_exit==15:
        min_age_exit='>10'
    if min_age_exit==8:
        min_age_exit='6-10'
    max_age_exit=max(age_exit)
    if max_age_exit==15:
        max_age_exit='>10'
    if max_age_exit==8:
        max_age_exit='6-10'
        
        
    age_value=list(age_exit_ser)
    age_index=list(age_exit_ser.index[:])
    for i in range(0,len(age_index)):
        if age_index[i]=='6-10':
            age_index[i]='8'
        elif age_index[i]=='>10':
            age_index[i]='15'
        else:
            continue
    sum_age= [a*int(b) for a,b in zip(age_value,age_index)]
    total_sum_age=sum(sum_age)
    avg_age_exit=total_sum_age/(df_coho['Exit'].value_counts()['yes']-
                    df_coho[df_coho['Exit']=='yes']['last_funding_status'].value_counts()['nan'])
    data.append(min_age_exit)
    data.append(max_age_exit)
    
    data.append(avg_age_exit)
    
    # State
    state_exit=df_coho[(df_coho['Exit']=='yes')&(df_coho['last_funding_status']!='nan')]['location'].value_counts()
    
    c=list(state_exit.index[0:5])
    d=list(state_exit[0:5])
    f=""
    dicti=dict(zip(c,d))
    for i,j in dicti.items():
        s="".join(str(i)+"-"+str(j))
        f=f+s+' | '
    data.append(f)
    workbook = xlsxwriter.Workbook('/home/ayesha/projects/notebooks/Fin.xlsx') 
    worksheet = workbook.add_worksheet() 
  

    row = 0
    column = 0

    content = ["Total No Of Companies", "Funded Companies Count", "Funded Companies Likelihood",  
                 "Exit Count","Exit Likelihood","Exit Age Min","Exit Age Max","Exit Age Avg","Exit Top Five States","Funded Companies Distribution",
               "Funded Companies Distribution Count","Funding Likelihood",
               "Funding type min age","Funding type max age","Funding type avg age","Funding Type state" 
               ] 

    # Heades
    for item in content : 

        # write operation perform 
        worksheet.write(row, column, item) 

        # incrementing the value of row by one 
        # with each iteratons. 
        column += 1
    
    # Single row , multiple columns entries
    column = 0
    for item in data : 
       
        worksheet.write(row +1, column, item) 

        column += 1
    
 
    # Multiple rows and single column 
    row=1    
    for item in final_funding_types:
        worksheet.write(row,column,item)
        row+=1
    
    
    row=1
    column+=1
    for item in last_funding_count:
        worksheet.write(row,column,item)
        row+=1
    
    row=1
    column+=1
    for item in funding_likelihood:
        worksheet.write(row,column,item)
        row+=1
    
    row=1
    column+=1
    for i in final_funding_types:
        
        df_i=df_coho[(df_coho['Exit']=='no')&(df_coho['last_funding_status']!='nan')&(df_coho['last_funding_status']==str(i))]
        i_age_ser=df_i['Age'].value_counts()
        i_age=list(df_i['Age'].unique())
        for i in range(0,len(i_age)):
            if i_age[i]=='6-10':
                i_age[i]='8'
            elif i_age[i]=='>10':
                i_age[i]='15'
            else:
                continue
        i_age = list(map(int, i_age))
        min_i_age=min(i_age)
        if min_i_age==15:
            min_i_age='>10'
        if min_i_age==8:
            min_i_age='6-10'
        worksheet.write(row,column,min_i_age)
        max_i_age=max(i_age)
        if max_i_age==15:
            max_i_age='>10'
        if max_i_age==8:
            max_i_age='6-10'
        worksheet.write(row,column+1,max_i_age)
        age_value=list(i_age_ser)
        age_index=list(i_age_ser.index[:])
        for i in range(0,len(age_index)):
            if age_index[i]=='6-10':
                age_index[i]='8'
            elif age_index[i]=='>10':
                age_index[i]='15'
            else:
                continue
        sum_age= [a*int(b) for a,b in zip(age_value,age_index)]
        total_sum_age=sum(sum_age)
        avg_age_exit=total_sum_age/(df_coho['Exit'].value_counts()['no']-
                        df_coho[df_coho['Exit']=='no']['last_funding_status'].value_counts()['nan'])
        worksheet.write(row,column+2,avg_age_exit)
        
        
        state_exit=df_i['location'].value_counts()
        c1=list(state_exit.index[0:5])
        d1=list(state_exit[0:5])
        f1=""
        dicti=dict(zip(c1,d1))
        #print(state_exit)
        for i1,j1 in dicti.items():
            s1="".join(str(i1)+"-"+str(j1))
            f1=f1+s1+' | '
            #print(i1)
        worksheet.write(row,column+3,f1)
        
        row+=1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    workbook.close() 
    
    

    return df_coho.shape
    


# In[761]:


state_age(industry_vertical='AIML')


# In[437]:


df_com[df_com['industry_vertical']=='AIML']['last_funding_status'].value_counts()


# In[594]:


age_exit=df_com[(df_com['Exit']=='yes')&(df_com['last_funding_status']!='nan')]['Age'].value_counts()
min_age_exit=age_exit[age_exit == min(age_exit)].index[0]


# In[679]:


age=list(df_com[(df_com['Exit']=='yes')&(df_com['last_funding_status']!='nan')&(df_com['last_funding_status']=='seed')]['Age'].unique())


# In[680]:


for i in range(0,len(age)):
    if age[i]=='6-10':
        age[i]='8'
    elif age[i]=='>10':
        age[i]='15'


# In[681]:


age = list(map(int, age))


# In[683]:


age.sort(reverse=True)


# In[684]:


age


# In[685]:


min(age)


# In[750]:


state_exit=df_com[(df_com['industry_vertical']=='AIML')&(df_com['Exit']=='no')&(df_com['last_funding_status']!='nan')&
                           (df_com['last_funding_status']=='series_c')]['location'].value_counts()


# In[751]:


c=list(state_exit.index[0:5])
d=list(state_exit[0:5])
f=""
dicti=dict(zip(c,d))
for i,j in dicti.items():
    s="".join(str(i)+"-"+str(j))
    f=f+s+' | '
print(f)


# In[705]:


type(f)


# In[616]:


max_age=max(age)
if max_age=='15':
    max_age='>10'
if max_age=='8':
    max_age='6-10'


# In[617]:


max_age


# In[550]:


a=df_com[(df_com['industry_vertical']=='AIML')&(df_com['Exit']=='yes')&(df_com['last_funding_status']!='nan')]['Age'].value_counts()


# In[572]:


x=list(a)


# In[573]:


y=list(a.index[:])


# In[574]:


for i in range(0,len(y)):
    if y[i]=='6-10':
        y[i]='8'
    elif y[i]=='>10':
        y[i]='15'
    else:
        continue


# In[570]:


y


# In[575]:


x


# In[578]:


summ= [a*int(b) for a,b in zip(x,y)]


# In[585]:


sum(summ)


# In[584]:


avg


# In[296]:


df_coho = df_com[(df_com['industry_vertical'] == 'AIML') & (df_com['no_of_employees'] == '1-10') & (df_com['location'] == 'NY')& (df_com['Age'] == '2')]

b = [i for i in df_coho['last_funding_status'].unique() if i!='nan']    
print(b)


# In[297]:


df_coho['last_funding_status'].value_counts()


# In[305]:


ser= df_coho['last_funding_status'].value_counts()
c=[]
for i in b:
    c.append(ser[i])


# In[306]:


print(c)


# In[14]:


import xlsxwriter 


# In[218]:


workbook = xlsxwriter.Workbook('/home/Ayesha/notebooks/Output.xlsx') 
worksheet = workbook.add_worksheet() 
  
# Start from the first cell. 
# Rows and columns are zero indexed. 
row = 0
column = 0
  
content = ["Total No Of Companies", "Funded Companies Count", "Funded Companies Likelihood", "Non Funded Companies Count", 
                    "Non Funded Companies Likelihood", "Type of Funding" , "Type of Funding Count","Type of Funding Likelihood", "Exit Count", "Exit Likelihood"] 
  
# iterating through content list 
for item in content : 
  
    # write operation perform 
    worksheet.write(row, column, item) 
  
    # incrementing the value of row by one 
    # with each iteratons. 
    column += 1
      
workbook.close() 


# In[4]:


df()


# In[5]:


df1.drop(columns = ['Exit','city' , 'Country'] ,inplace = True)


# In[6]:


df1.rename(columns = {'investment_type' : 'last_funding_status'} , inplace = True)


# In[7]:


df1.head()


# In[8]:


df1.drop(columns = ['company_name'] , inplace = True)


# In[9]:


df1.head()


# In[ ]:





# In[10]:


df1.rename(columns = {'state' : 'location'} , inplace = True)


# In[11]:


df1.rename(columns = {'funding_total' : 'total_funding'} , inplace = True)


# In[12]:


df1.rename(columns = {'founders_prev_startup_exp' : 'entrepreneurial_experience'} , inplace = True)


# In[13]:


df1.head()


# In[39]:


df1['last_funding_status'] = df1['last_funding_status'].fillna('nan')


# In[40]:


df1['funding_status'] = df1['last_funding_status'].apply(lambda x : 'no' if x=='nan' else 'yes')


# In[41]:


df1.to_excel('/home/Ayesha/notebooks/df1_excel.xlsx' , index = False) 


# In[33]:


#df1['location'].value_counts()


# In[191]:


df1_a = df1[(df1['industry_vertical']== 'AIML')  & (df1['no_of_employees']=='1-10') & (df1['Age']== '3') & (df1['location']=='NY')]


# In[192]:


df1_a.shape


# In[193]:


df1_a['last_funding_status'].value_counts()


# In[139]:


df1['no_of_employees'].value_counts()


# In[194]:


def cohort(ind_vr, no_emp , loc , age) :
    df_coho = df1[(df1['industry_vertical'] == ind_vr) & (df1['no_of_employees'] == no_emp) & (df1['location'] == loc)& (df1['Age'] == age)]
    return df_coho['last_funding_status'].value_counts()


# In[195]:


cohort('AIML' , '1-10' , 'NY' ,'3')


# In[ ]:


total


# In[2]:


df1.head()


# In[220]:


df1.shape


# In[185]:


pd.set_option("display.max_columns", 130)


# In[45]:


import numpy as np


# In[224]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(df1['funding_status'])


# In[222]:


# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown = 'ignore')


# In[112]:


# def OneHot(x) :
#     label = [] 
#     for i in df1[x]:
#         label.append(i)
#     ar = np.asarray(label)
#     ar = ar.reshape(48971,1)
#     enc.fit(ar)
#     l = enc.transform(ar)
#     return(l)


# In[298]:


df1['funding_status'].value_counts()


# In[294]:


print((y == 1).sum())


# In[291]:


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# In[234]:


df_with_dummies = pd.get_dummies( df1, columns = ['industry_vertical', 'Age',
       'location', 'no_of_employees', 'type_of_startup',
       'entrepreneurial_experience', 'founders_prev_fundraising_exp',
       'previous_exit_of_founders', 'previous_exits_by_lead_investors',
       'success_rate_of_lead_investors', 'revenue_model', 'knowledge_support',
       'funding_status'] )


# In[235]:


df_with_dummies.drop(columns=['funding_status_no','funding_status_yes'],inplace=True)


# In[236]:


df_with_dummies.drop(columns=['total_funding','last_funding_status'],inplace=True)


# In[237]:


df_with_dummies.head()


# In[238]:


df_with_dummies.shape


# In[286]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_with_dummies, y, test_size=0.3,stratify=y, random_state=0)


# In[470]:


x_test.shape


# In[473]:


x_test[(x_test['industry_vertical_healthcare']==1)].shape


# In[ ]:





# ## Logistic Regression

# In[433]:


from sklearn.linear_model import LogisticRegression


# In[434]:


logisticRegr = LogisticRegression()


# In[435]:


logisticRegr.fit(x_train, y_train)


# In[436]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score


# In[437]:


yscore = logisticRegr.predict(x_test)
precision = precision_score(y_test, yscore)
recall = recall_score(y_test, yscore)
acc = logisticRegr.score(x_test , y_test)

print(precision)
print(recall)
print(acc)


# In[438]:


temp=logisticRegr.predict_proba(x_test)


# In[439]:


t = [x[1] for x in temp.tolist()]


# In[440]:


x_test['probs']=t


# In[441]:


x_test[['probs','industry_vertical_healthcare','industry_vertical_Fintech','industry_vertical_AIML']]


# In[442]:


df_hProb=x_test[(x_test['industry_vertical_healthcare']==1) & (x_test['probs']>0.5)].head(5)


# In[443]:


df_hProb


# In[444]:


df_h1=df1.iloc[[6577,2030,4293,3065,3360],:]


# In[445]:


df_h1['Probability']=df_hProb['probs']


# In[446]:


df_h1


# In[ ]:





# In[447]:


df_hProb=x_test[(x_test['industry_vertical_AIML']==1) & (x_test['probs']>0.5)].head(5)


# In[448]:


df_hProb


# In[449]:


df_a1=df1.iloc[[19627,21156,21917,22435,21819],:]


# In[450]:


df_a1['Probability']=df_hProb['probs']


# In[451]:


df_a1


# In[ ]:





# In[452]:


df_hProb=x_test[(x_test['industry_vertical_Fintech']==1) & (x_test['probs']>0.5)].head(5)


# In[453]:


df_hProb


# In[454]:


df_f1=df1.iloc[[32709,33903,34225],:]


# In[455]:


df_f1['Probability']=df_hProb['probs']


# In[456]:


df_f1


# In[457]:


df_logistic_regression=pd.concat([df_h1,df_f1,df_a1])


# In[458]:


df_logistic_regression.head(20)


# In[459]:


df_logistic_regression.to_csv('logistic_regression_probabilty.csv',index=False)


# In[460]:


df_logistic_regression.to_excel('logistic_regression_probabilty.xlsx',index=False)


# In[461]:


x_test.drop(columns=['probs'],inplace=True)


# In[462]:


x_test.shape


# ## Decision Tree

# In[402]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# In[403]:


clf = DecisionTreeClassifier()


# In[404]:


clf.fit(x_train, y_train)


# In[405]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score ,classification_report


# In[407]:


yscore = clf.predict(x_test)
precision = precision_score(y_test, yscore)
recall = recall_score(y_test, yscore)
acc = clf.score(x_test , y_test)

print(precision)
print(recall)
print(acc)


# In[408]:


temp=clf.predict_proba(x_test)


# In[409]:


t = [x[1] for x in temp.tolist()]


# In[410]:


x_test['probs']=t


# In[411]:


x_test[['probs','industry_vertical_healthcare','industry_vertical_Fintech','industry_vertical_AIML']]


# In[412]:


df_hProb=x_test[(x_test['industry_vertical_healthcare']==1) & (x_test['probs']>0.5)].head(5)


# In[413]:


df_hProb


# In[414]:


df_h1=df1.iloc[[333,2006,14298,9691,3413],:]


# In[415]:


df_h1['Probability']=df_hProb['probs']


# In[416]:


df_h1


# In[ ]:





# In[417]:


df_hProb=x_test[(x_test['industry_vertical_AIML']==1) & (x_test['probs']>0.5)].head(5)


# In[418]:


df_hProb


# In[419]:


df_a1=df1.iloc[[23469,19682,21070,20342,25162],:]


# In[420]:


df_a1['Probability']=df_hProb['probs']


# In[421]:


df_a1


# In[ ]:





# In[422]:


df_hProb=x_test[(x_test['industry_vertical_Fintech']==1) & (x_test['probs']>0.5)].head(5)


# In[423]:


df_hProb


# In[424]:


df_f1=df1.iloc[[36264,30832,41697,30953,34902],:]


# In[425]:


df_f1['Probability']=df_hProb['probs']


# In[426]:


df_f1


# In[427]:


df_decision_tree=pd.concat([df_h1,df_f1,df_a1])


# In[428]:


df_decision_tree.head(20)


# In[429]:


df_decision_tree.to_csv('decision_tree_probabilty.csv',index=False)


# In[430]:


df_decision_tree.to_excel('decision_tree_probabilty.xlsx',index=False)


# In[431]:


x_test.drop(columns=['probs'],inplace=True)


# In[432]:


x_test.shape


# ## KNN

# In[373]:


from sklearn.neighbors import KNeighborsClassifier


# In[374]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[375]:


knn.fit(x_train, y_train)


# In[376]:


yscore = knn.predict(x_test)
precision = precision_score(y_test, yscore)
recall = recall_score(y_test, yscore)
acc = knn.score(x_test , y_test)

print(precision)
print(recall)
print(acc)


# In[377]:


temp=knn.predict_proba(x_test)


# In[378]:


t = [x[1] for x in temp.tolist()]


# In[379]:


x_test['probs']=t


# In[380]:


x_test[['probs','industry_vertical_healthcare','industry_vertical_Fintech','industry_vertical_AIML']]


# In[381]:


df_hProb=x_test[(x_test['industry_vertical_healthcare']==1) & (x_test['probs']>0.5)].head(5)


# In[382]:


df_hProb


# In[383]:


df_h1=df1.iloc[[9374,2620,14298,6577,4555],:]


# In[384]:


df_h1['Probability']=df_hProb['probs']


# In[385]:


df_h1


# In[ ]:





# In[386]:


df_hProb=x_test[(x_test['industry_vertical_AIML']==1) & (x_test['probs']>0.5)].head(5)


# In[387]:


df_hProb


# In[388]:


df_a1=df1.iloc[[19456,25162,19049,24372,26371],:]


# In[389]:


df_a1['Probability']=df_hProb['probs']


# In[390]:


df_a1


# In[ ]:





# In[391]:


df_hProb=x_test[(x_test['industry_vertical_Fintech']==1) & (x_test['probs']>0.5)].head(5)


# In[392]:


df_hProb


# In[393]:


df_f1=df1.iloc[[34902,33305,48809,33712,31209],:]


# In[394]:


df_f1['Probability']=df_hProb['probs']


# In[395]:


df_f1


# In[396]:


df_KNN=pd.concat([df_h1,df_f1,df_a1])


# In[397]:


df_KNN.head(20)


# In[398]:


df_KNN.to_csv('KNN_probability.csv',index=False)


# In[399]:


df_KNN.to_excel('KNN_probability.xlsx',index=False)


# In[400]:


x_test.drop(columns=['probs'],inplace=True)


# In[401]:


x_test.shape


# ## SVM

# In[287]:


from sklearn import svm


# In[288]:


clf = svm.SVC(kernel='poly')


# In[289]:


clf.fit(x_train, y_train)


# In[290]:


yscore = clf.predict(x_test)
precision = precision_score(y_test, yscore)
recall = recall_score(y_test, yscore)
acc = clf.score(x_test , y_test)

print(precision)
print(recall)
print(acc)


# In[272]:


print(classification_report(y_test,yscore))


# ## Random Forest

# In[313]:


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)

yscore=clf.predict(x_test)
precision = precision_score(y_test, yscore)
recall = recall_score(y_test, yscore)
acc = clf.score(x_test , y_test)

print(precision)
print(recall)
print(acc)


# In[308]:


yscore


# In[314]:


temp=clf.predict_proba(x_test)


# In[315]:


t = [x[1] for x in temp.tolist()]


# In[328]:


x_test['probs']=t


# In[329]:


x_test[['probs','industry_vertical_healthcare','industry_vertical_Fintech','industry_vertical_AIML']]


# In[339]:


df_hProb=x_test[(x_test['industry_vertical_healthcare']==1) & (x_test['probs']>0.5)].head(5)


# In[340]:


df_hProb


# In[346]:


df_h1=df1.iloc[[14298,6577,11611,7239,14466],:]


# In[347]:


df_h1['Probability']=df_hProb['probs']


# In[348]:


df_h1


# In[ ]:





# In[354]:


df_hProb=x_test[(x_test['industry_vertical_AIML']==1) & (x_test['probs']>0.5)].head(5)


# In[355]:


df_hProb


# In[356]:


df_a1=df1.iloc[[19682,25162,26371,28871,22941],:]


# In[357]:


df_a1['Probability']=df_hProb['probs']


# In[358]:


df_a1


# In[ ]:





# In[359]:


df_hProb=x_test[(x_test['industry_vertical_Fintech']==1) & (x_test['probs']>0.5)].head(5)


# In[360]:


df_hProb


# In[361]:


df_f1=df1.iloc[[36264,30953,34902,45158,31278],:]


# In[362]:


df_f1['Probability']=df_hProb['probs']


# In[363]:


df_f1


# In[365]:


df_random_forest=pd.concat([df_h1,df_f1,df_a1])


# In[366]:


df_random_forest.head(20)


# In[367]:


df_random_forest.to_csv('random_forest_probabilty.csv',index=False)


# In[368]:


df_random_forest.to_excel('random_forest_probabilty.xlsx',index=False)


# In[371]:


x_test.drop(columns=['probs'],inplace=True)


# In[372]:


x_test.shape


# In[ ]:





# In[1]:


import pymongo


# In[51]:


from pymongo import MongoClient
client = MongoClient()


# In[47]:


dbms = client['company']


# In[66]:


col = dbms['dataframe']


# In[53]:


import json


# In[54]:


df_com = pd.read_excel('final_fintech_healthcare_aiml_companies.xlsx')


# In[55]:


df_com.shape


# In[67]:


records = json.loads(df_com.to_json()).values()
db.col.insert(records)


# In[63]:


db.list_collection_names()


# In[68]:


x = list(col.find())


# In[69]:


len(x)


# In[70]:


for x in col.find():
    print(x)


# In[ ]:





# In[279]:


df_com.head()


# In[280]:


df_com.shape


# In[281]:


df_com[df_com['company_name'] == 'PatientPing']


# In[286]:


df_com[(df_com['industry_vertical'] == 'healthcare') & (df_com['Country'] == 'USA')]


# In[ ]:




