#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astroquery.mast import Observations
f115w_obs = Observations.query_criteria(provenance_name="jades",instrument_name='NIRCAM/IMAGE',filters='F115W')
data_products = Observations.get_product_list(f115w_obs)
Observations.download_products(data_products)


# In[ ]:




