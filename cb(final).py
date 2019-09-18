#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


df=pd.read_csv('/home/ayesha/projects/notebooks/organizations.csv')


# In[5]:


df.head()


# In[3]:


df[df['company_name']=='Element']


# In[6]:


df_funding=pd.read_csv('/home/ayesha/projects/notebooks/funding_info_all.csv')


# In[7]:


df_funding.head()


# In[22]:


df[df['company_name']=='PatientPing']


# In[14]:


df_funding[df_funding['company_name']=='PatientPing']


# In[ ]:





# In[6]:


pd.set_option('display.max_columns', None)


# In[7]:


df.shape


# In[8]:


df =df.loc[df['country_code']=='USA']


# In[9]:


df_funding=df_funding.loc[df_funding['country_code_x']=='USA']


# In[10]:


df.shape


# In[13]:


df =df.loc[df['status']!='closed']


# In[14]:


df.shape[0]


# In[15]:


df.reset_index(drop=True, inplace=True)


# In[17]:


df_new = df[df['category_list'].notnull() & df['founded_on'].notnull()]


# In[18]:


df_new.reset_index(drop=True, inplace=True)


# In[19]:


df_new.shape


# In[20]:


df_new['location']=df_new['country_code']+' '+df_new['state_code']+' '+ df_new['city']


# In[21]:


df_new['total_funding']=df_new['funding_total_usd']/(10**6)


# In[22]:


import numpy as np


# In[23]:


df_new['is_ipo'] = np.where(df_new['status']=='ipo', '1', '0')


# In[24]:


df_new['is_M&A'] = np.where(df_new['status']=='acquired', '1', '0')


# In[33]:


import datetime
from dateutil import relativedelta
import datetime
now = datetime.datetime.now()


# In[26]:


df_new=df_new.assign(age_in_months='')


# In[ ]:





# In[27]:


df_new['founded_on']=pd.to_datetime(df_new['founded_on'],errors='coerce')


# In[28]:


df_new['last_funding_on']=pd.to_datetime(df_new['last_funding_on'],errors='coerce')


# In[29]:


df_new['age_in_months']=df_new['last_funding_on'].subtract(df_new['founded_on'])


# In[30]:


s=df_new[df_new['age_in_months']==pd.Timedelta(0,'D')] #n,s


# In[31]:


n=df_new[df_new['age_in_months']<pd.Timedelta(0,'D')]


# In[33]:


n[n['company_name']=='BuroHQ']


# In[35]:


df_funding[df_funding['company_name'].isin(n['company_name'])].drop_duplicates(subset=['company_name'],keep='first')


# In[36]:


df_AIML = df_new.copy(deep=True)


# In[37]:


df_fintech=df_new.copy(deep=True)


# In[38]:


df_healthcare = df_new.copy(deep=True)


# In[39]:


df_healthcare = pd.DataFrame().reindex_like(df_new)


# In[40]:


df_AIML = pd.DataFrame().reindex_like(df_new)


# In[41]:


df_fintech = pd.DataFrame().reindex_like(df_new)


# In[42]:


df_healthcare.shape


# In[43]:


healthcare=['Medical','First Aid','Assistive Technology','Cosmetic Surgery','Fertility','Medical Device','Nursing and Residential Care','Clinical Trials','Alternative Medicine','Elderly','Dental','mHealth','Home Health Care','Personal Health','Assisted Living','Emergency Medicine','Hospital','Health Diagnostics','Electronic Health Record (EHR)','Outpatient Care','Elder Care','Wellness','Diabetes','Baby','Therapeutics']
AIML = ['Artificial Intelligence','E-Learning','Data Visualization','Facial Recognition','Prediction Markets','Spam Filtering','Speech Recognition','Marketing Automation','Text Analytics','Smart Home','Natural Language Processing','Augmented Reality','Predictive Analytics','Big Data','Data Mining','Autonomous Vehicles','Image Recognition','Fraud Detection','Analytics','Robotics','Semantic Search','Intelligent Systems','Business Intelligence','Machine Learning','Computer Vision','Smart Building']


# In[44]:


fintech=['Payments','Prediction Markets','Blockchain','Personal Finance','Financial Exchange','Finance','Financial Services','Mobile Payments','FinTech','Insurance','Banking']


# In[45]:


cat_string = "|".join(healthcare)
df_new['category_list'] = df_new['category_list'].fillna('')
df_healthcare= df_new[df_new['category_list'].str.contains(cat_string)]


# In[46]:


df_healthcare.shape


# In[47]:


cat_string = "|".join(AIML)
df_new['category_list'] = df_new['category_list'].fillna('')
df_AIML= df_new[df_new['category_list'].str.contains(cat_string)]


# In[48]:


df_AIML.shape


# In[49]:


cat_string = "|".join(fintech)
df_new['category_list'] = df_new['category_list'].fillna('')
df_fintech= df_new[df_new['category_list'].str.contains(cat_string)]


# In[50]:


df_fintech.shape


# In[51]:


df_AIML['investment_type']=''
df_healthcare['investment_type']=''
df_fintech['investment_type']=''


# In[52]:


df_healthcare.reset_index(drop=True, inplace=True)
df_AIML.reset_index(drop=True, inplace=True)
df_fintech.reset_index(drop=True, inplace=True)


# In[53]:


import warnings
warnings.filterwarnings('ignore')


# In[55]:


df_funding['announced_on']=pd.to_datetime(df_funding['announced_on'],errors='coerce')


# In[56]:


df_funding.sort_values(by = ["announced_on"],ascending=False,inplace=True)


# In[57]:


df_funding.drop_duplicates(subset=['company_name'],keep='first',inplace=True)


# In[58]:


dict_investment = dict(zip(df_funding['company_name'], df_funding['investment_type']))


# In[59]:


dict_investment


# In[60]:


df_AIML['investment_type']=df_AIML.apply(lambda x: dict_investment[x['company_name']] if x['funding_rounds']>0 else 'nan',axis=1 )


# In[61]:


df_healthcare['investment_type']=df_healthcare.apply(lambda x: dict_investment[x['company_name']] if x['funding_rounds']>0 else 'nan',axis=1 )


# In[62]:


df_fintech['investment_type']=df_fintech.apply(lambda x: dict_investment[x['company_name']] if x['funding_rounds']>0 else 'nan',axis=1 )


# In[63]:


df_AIML['investment_type'].unique()


# In[64]:


df_healthcare['industry_vertical']='healthcare'


# In[65]:


df_AIML['industry_vertical']='AIML'


# In[66]:


df_fintech['industry_vertical']='Fintech'


# In[67]:


df_healthcare['location']=df_healthcare['country_code']+' '+df_healthcare['state_code']+' '+ df_healthcare['city']


# In[68]:


df_AIML['location']=df_AIML['country_code']+' '+df_AIML['state_code']+' '+ df_AIML['city']


# In[69]:


df_fintech['location']=df_fintech['country_code']+' '+df_fintech['state_code']+' '+ df_fintech['city']


# In[70]:


df_healthcare = df_healthcare[['company_name','total_funding','is_ipo','is_M&A','age_in_months','location','investment_type','industry_vertical']]


# In[71]:


df_AIML = df_AIML[['company_name','total_funding','is_ipo','is_M&A','age_in_months','location','investment_type','industry_vertical']]


# In[72]:


df_fintech = df_fintech[['company_name','total_funding','is_ipo','is_M&A','age_in_months','location','investment_type','industry_vertical']]


# In[73]:


df_AIML.shape


# In[74]:


df_healthcare['time_to_LFS']=df_healthcare["age_in_months"].dt.days/365


# In[75]:


df_AIML['time_to_LFS']=df_AIML["age_in_months"].dt.days/365


# In[76]:


df_fintech['time_to_LFS']=df_fintech["age_in_months"].dt.days/365


# In[77]:


df_healthcare.head()


# In[78]:


df_healthcare['is_ipo']=df_healthcare['is_ipo'].apply(lambda x: bool(int(x)))


# In[79]:


df_AIML['is_ipo']=df_AIML['is_ipo'].apply(lambda x: bool(int(x)))


# In[80]:


df_fintech['is_ipo']=df_fintech['is_ipo'].apply(lambda x: bool(int(x)))


# In[81]:


df_healthcare.head()


# In[82]:


df_healthcare['is_M&A']=df_healthcare['is_M&A'].apply(lambda x: bool(int(x)))


# In[83]:


df_AIML['is_M&A']=df_AIML['is_M&A'].apply(lambda x: bool(int(x)))


# In[84]:


df_fintech['is_M&A']=df_fintech['is_M&A'].apply(lambda x: bool(int(x)))


# In[85]:


df_fintech.head()


# In[86]:


df_healthcare['Exit']=df_healthcare['is_ipo'] | df_healthcare['is_M&A']


# In[87]:


df_AIML['Exit']=df_AIML['is_ipo'] | df_AIML['is_M&A']


# In[88]:


df_fintech['Exit']=df_fintech['is_ipo'] | df_fintech['is_M&A']


# In[89]:


df_healthcare.head()


# In[90]:


df_com = pd.concat([df_healthcare, df_AIML,df_fintech])


# In[91]:


df_com['total_funding'] = df_com['total_funding'].fillna(0)


# In[92]:


df_com['funding_total'] = ['undisclosed' if x==0 else '<1' if x <=1 else '1-10' if 1<x<=10 else '10-50' if 10<x<=50 else '50-100' if 50<x<=100 else '100+' for x in df_com['total_funding']]


# In[93]:


df_com.head()


# In[94]:


dict_emp = dict(zip(df['company_name'], df['employee_count']))


# In[95]:


dict_emp


# In[96]:


dict_state = dict(zip(df['company_name'], df['state_code']))


# In[97]:


df_com['state']=df_com['company_name'].map(dict_state)


# In[98]:


df_com['no_of_employees']=df_com['company_name'].map(dict_emp)


# In[99]:


df_com.head()


# In[100]:


dict_city = dict(zip(df['company_name'], df['city']))


# In[101]:


df_com['city']=df_com['company_name'].map(dict_city)


# In[102]:


df_com.head()


# In[103]:


df_com.drop(columns=['is_ipo', 'is_M&A','age_in_months','location'],inplace=True)


# In[104]:


df_com.head()


# In[105]:


dict_country = dict(zip(df['company_name'], df['country_code']))


# In[106]:


df_com['Country']=df_com['company_name'].map(dict_country)


# In[108]:


df_com.shape


# In[109]:


mylist = ["B2B", "B2C"]
import random


# In[110]:


df_com['type_of_startup']=random.choices(mylist, weights = [30, 70], k = 48971)


# In[111]:


founders=['yes','no']


# In[112]:


success = ['0%' , '0-5%' , '5-10%', '10+%']


# In[113]:


revenue = ['commission-based' , 'fee-for-service' , 'Advertising', 'SaaS' ,'Other']


# In[114]:


df_com['founders_prev_startup_exp']=random.choices(founders, weights = [5, 95], k = 48971)


# In[115]:


df_com['founders_prev_fundraising_exp']=random.choices(founders, weights = [5, 95], k = 48971)


# In[116]:


df_com['previous_exit_of_founders'] = random.choices(founders , weights = [15 , 85] , k = 48971)


# In[117]:


df_com['previous_exits_by_lead_investors'] = random.choices(founders , weights = [10 , 90] , k = 48971)


# In[118]:


df_com['success_rate_of_lead_investors'] = random.choices(success , weights = [72 , 14, 9, 5] , k = 48971)


# In[119]:


df_com['revenue_model'] = random.choices(revenue , weights = [5 , 5, 10, 70, 10] , k = 48971)


# In[120]:


dict_investor = dict(zip(df_funding['company_name'],df_funding['investor_type']))


# In[121]:


dict_investor


# In[122]:


df_com['knowledge_support'] = df_com['company_name'].map(dict_investor)


# In[123]:


req_investor = ['accelerator','incubator','university_program']


# In[124]:


df_com['knowledge_support_bool'] = df_com['knowledge_support'].apply(lambda x : 'yes' if x in req_investor else 'no')


# In[125]:


df_com.drop(columns = ['knowledge_support'] , inplace= True)


# In[126]:


df_com.rename(columns = {'knowledge_support_bool' : 'knowledge_support'} , inplace = True)


# In[127]:


df_com.shape


# In[128]:


df_com['Exit']=df_com['Exit'].apply(lambda x: 'yes' if x else 'no')


# In[129]:


df_com.head()


# In[16]:


#importing final excel file to df_com dataframe to make any changes.
import pandas as pd
df_com=pd.read_excel('/home/ayesha/projects/notebooks/final_fintech_healthcare_aiml_companies.xlsx')


# In[32]:


df_com.head()


# In[17]:


df_com[df_com['company_name']=='PatientPing']


# In[18]:


df_com[(df_com['total_funding']== 41.2) & (df_com['Country'] == 'USA')]


# In[31]:


df_com[(df_com['total_funding'] == 41.2) & (df_com['industry_vertical'] == 'healthcare')]


# In[21]:


df_com['Age'][df_com['company_name'] == 'HealthcareSource']


# In[7]:


df_h=df_com[df_com['industry_vertical']=='healthcare']


# In[8]:


df_h.shape


# In[259]:


df_h.head(100)


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


# In[130]:


df_com.head()


# In[131]:


df_com.drop(columns = ['total_funding'] , inplace = True)


# In[132]:


df_com.shape


# In[134]:


df_com.rename(columns = {'state' : 'location' } , inplace = True)


# In[135]:


df_com.rename(columns = {'investment_type' : 'last_funding_status' } , inplace = True)


# In[136]:


df_com['last_funding_status'] = df_com['last_funding_status'].fillna('nan')


# In[137]:


df_com.to_excel('final_fintech_healthcare_aiml_companies.xlsx' , index = False)


# In[138]:


df_com[(~df_com['time_to_LFS'].notnull())&(df_com['last_funding_status']!='nan')]


# In[140]:


df_com[(df_com['funding_total']=='undisclosed') &(df_com['last_funding_status']!='nan')]


# In[141]:


df_com[df_com['time_to_LFS']>0].groupby(['industry_vertical']).describe()


# In[142]:


df_funding_no_drop=pd.read_csv('/home/kartikey/Documents/fintech_healthcare_ai/funding_info_all.csv')


# In[143]:


df_funding_no_drop.sort_values(by = ["announced_on"],ascending=True,inplace=True)


# In[144]:


# this is dropping rows which dont have seed, angel or series_a


# In[145]:


df_funding_no_drop=df_funding_no_drop[df_funding_no_drop['investment_type'].isin(['seed','angel','series_a'])]


# In[146]:


df_funding_no_drop=df_funding_no_drop[df_funding_no_drop['company_name'].isin(df_com[df_com['industry_vertical']=='AIML']['company_name'])]


# In[161]:


df_funding_no_drop


# In[237]:


df_funding_no_drop[df_funding_no_drop['company_name']=='10XTS']


# In[147]:


pd.options.display.max_colwidth = 100


# In[148]:


state_dist=df_com.groupby('last_funding_status')['location'].value_counts()


# In[149]:


time_dist=df_com.groupby('last_funding_status')['time_to_LFS'].describe()


# In[153]:


reqd=['angel','seed','series_a']
temp=state_dist[reqd].groupby(level=0).nlargest()
temp['angel']


# In[154]:


time_dist.loc[reqd,['min','mean','max']]


# In[155]:


# this has first date and its corresponding funding, we just need to subtract different dates


# In[156]:


date_and_fund=df_funding_no_drop[~df_funding_no_drop.duplicated(subset=['company_name','investment_type'],keep='first')].groupby('company_name').apply(lambda x: dict(zip(x['announced_on'],x['investment_type']))).reset_index().rename(columns={0:'date_and_funding'})


# In[157]:


filtered=date_and_fund[date_and_fund['date_and_funding'].apply(lambda x : set(['series_a','seed']).issubset(x.values()))]


# In[158]:


filtered


# In[159]:


from datetime import datetime
def f(x,a,b):
    p=dict(zip(x.values(),x.keys()))
    if p[a]<p[b]:
        diff=datetime.strptime(p[b],'%Y-%m-%d')-datetime.strptime(p[a],'%Y-%m-%d')
    else:
        diff=0
    return diff


# In[160]:


filtered['date_and_funding'].apply(lambda x : f(x,'seed','series_a') ).reset_index()


# In[150]:


def cohort(**kwargs) :

    
    filters=True
    for key, value in kwargs.items(): 
        filters=filters&(df_com[key]==value)
    df_coho = df_com[filters] 
    data = []
#     data.extend([df_coho.shape[0] , df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'] , 
#                  (df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])*100/df_coho.shape[0], 
#                  df_coho['last_funding_status'].value_counts()['nan'] , 
#                  (df_coho['last_funding_status'].value_counts()['nan']*100/df_coho.shape[0]) , 
#                  df_coho['Exit'].value_counts()['yes'] , 
#                  (df_coho['Exit'].value_counts()['yes']*100/df_coho.shape[0]) ,
#                 df_coho['last_funding_status'].])
    
    
    data.append(df_coho.shape[0])
    try :
        data.append(df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])
    except:
        data.append(df_coho.shape[0])
    try :
        data.append((df_coho.shape[0]-df_coho['last_funding_status'].value_counts()['nan'])*100/df_coho.shape[0])
    except:
        data.append(100)
    try:
        data.append(df_coho['last_funding_status'].value_counts()['nan'])
    except:
        data.append(0)
    try:
        data.append((df_coho['last_funding_status'].value_counts()['nan']*100/df_coho.shape[0]))
    except:
        data.append(0)
    try:
        data.append(df_coho['Exit'].value_counts()['yes'])
    except:
        data.append(0)
    try:
        data.append((df_coho['Exit'].value_counts()['yes']*100/df_coho.shape[0]))
    except:
        data.append(0)
    last_funding= [i for i in df_coho['last_funding_status'].unique() if i!='nan'] 
    ser=df_coho['last_funding_status'].value_counts()
    
    last_funding_count=[]
    
    for i in last_funding:
        last_funding_count.append(ser[i])
    
    funding_likelihood=[(i*100)/df_coho.shape[0] for i in last_funding_count]
    
    
    
    workbook = xlsxwriter.Workbook('Fin.xlsx') 
    worksheet = workbook.add_worksheet() 
  

    row = 0
    column = 0

    content = ["Total No Of Companies", "Funded Companies Count", "Funded Companies Likelihood", 
               "Non Funded Companies Count", 
                        "Non Funded Companies Likelihood", 
                 "Exit Count","Exit Likelihood","Funded Companies Distribution",
               "Funded Companies Distribution Count","Funding Likelihood", 
               ] 

    # iterating through content list 
    for item in content : 

        # write operation perform 
        worksheet.write(row, column, item) 

        # incrementing the value of row by one 
        # with each iteratons. 
        column += 1
    
    column = 0
    for item in data : 
        # write operation perform 
        worksheet.write(row +1, column, item) 

        # incrementing the value of row by one 
        # with each iteratons. 
        column += 1
    
    row=1    
    for item in last_funding:
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
    
    
    workbook.close() 
    
    

    return df_coho.head()
    


# In[313]:


cohort('AIML' , '1-10' , 'NY','2')


# In[319]:


cohort('Fintech' , '101-250' , 'TX','5')


# In[315]:


cohort('healthcare' , '11-50' , 'MA','4')


# In[317]:


cohort('AIML' , '1-10' , 'FL','1')


# In[18]:


def samplefunc(**kwargs):
    filters=True
    for key, value in kwargs.items(): 
        filters=filters&(df_com[key]==value)
    df_coho = df_com[filters] 

    return df_coho.head()


# In[ ]:





# In[155]:


cohort(industry_vertical='AIML',no_of_employees='1-10')


# In[9]:


df_coho = df_com[(df_com['industry_vertical'] == 'AIML') & (df_com['no_of_employees'] == '1-10') & (df_com['location'] == 'NY')& (df_com['Age'] == '2')]

b = [i for i in df_coho['last_funding_status'].unique() if i!='nan']    
print(b)


# In[10]:


df_coho['last_funding_status'].value_counts()


# In[305]:


ser= df_coho['last_funding_status'].value_counts()
c=[]
for i in b:
    c.append(ser[i])


# In[306]:


print(c)


# In[153]:


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


# In[2]:


from pymongo import MongoClient
import pandas as pd
client = MongoClient('192.168.1.28', 27017)
client.admin.authenticate('admin', 'athena', mechanism = 'SCRAM-SHA-1')
#db = client['Autophrase']
#collection = db['predicted_phrases']
#client = MongoClient()
db=client.test
employee = db.employee
df = pd.read_csv("/home/kartikey/Documents/fintech_healthcare_ai/organizations.csv") #csv file which you want to import
records_ = df.to_dict(orient = 'records')
result = db.employee.insert_many(records_ )


# In[34]:


df_com.head()


# In[ ]:


df_com['founded_on']=pd.to_datetime(df_com['founded_on'],errors='coerce')


# In[ ]:





# In[35]:


df_latest = df_funding.copy()


# In[38]:


import datetime
from dateutil import relativedelta
import datetime
now = datetime.datetime.now()


# In[39]:


df_latest['founded_on']=df_latest['founded_on'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d') )


# In[ ]:


df_latest['age_in_months']=df_latest['founded_on'].apply(lambda x:12*relativedelta.relativedelta(now,x ).years+relativedelta.relativedelta(now,x ).months )

