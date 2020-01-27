def gmm_customer_scoring(data):
    #Import Libraries
    import pandas as pd
    import datetime
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:90% !important; }</style>"))
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler
    
   
    #Customer frequency with more than 2 transactions
    frequency_customer = data.pivot_table(index='customer_id', aggfunc={'PaymentTransactionId': 'count'}).reset_index().rename(columns={'PaymentTransactionId': 'frequency'})
    #customer which has more than 2 transactions
    more_than_2_transaction_customers = list(frequency_customer[frequency_customer['frequency'] >1]['customer_id'])
    #merge with the real data
    data = data.query("customer_id in @more_than_2_transaction_customers").reset_index(drop=True)
    
    ##########RFM############ & Mean, Min, Max, Std
    #recency
    current_date = max(data['Created_Time']) + datetime.timedelta(days=1)
    recency_merchant = data.pivot_table(index='customer_id', aggfunc={'Created_Time': 'max'}).reset_index().rename(columns={'Created_Time': 'max_date'})
    recency_merchant['recency'] = recency_merchant.apply(lambda row: (current_date - row['max_date']).total_seconds() / 60 / 60, axis=1)
    
    #frequency
    data['Created_Time_prev'] = data.sort_values(['Created_Time', 'customer_id'], ascending=True).groupby('customer_id')['Created_Time'].shift(1)
    data['hour_diff'] = data.apply(lambda row: (row['Created_Time'] - row['Created_Time_prev']).total_seconds() / 60 / 60 ,axis=1) 
    data['hour_diff' + '_mean'], data['hour_diff' + '_std'], data['hour_diff' + '_count'] = data['hour_diff'], data['hour_diff'], data['PaymentTransactionId']
    frequency_merchant = data.pivot_table(index='customer_id', aggfunc={'hour_diff' + '_mean': 'mean', 'hour_diff' + '_std': 'std', 'hour_diff' + '_count': 'count'}).reset_index().rename(columns={"hour_diff_mean": 'frequency'})
    
    #monetary Median
    monetary_merchant = data.pivot_table(index='customer_id', aggfunc={'Amount': 'median'}).reset_index().rename(columns={'Amount': 'monetary'})
    
    #mean
    monetary_merchant_mean = data.pivot_table(index='customer_id', aggfunc={'Amount': 'mean'}).reset_index().rename(columns={'Amount': 'monetary_mean'})
    
    #min
    monetary_merchant_min = data.pivot_table(index='customer_id', aggfunc={'Amount': 'min'}).reset_index().rename(columns={'Amount': 'monetary_min'})
    
    #max
    monetary_merchant_max = data.pivot_table(index='customer_id', aggfunc={'Amount': 'max'}).reset_index().rename(columns={'Amount': 'monetary_max'})
    
    #std
    monetary_merchant_std = data.pivot_table(index='customer_id', aggfunc={'Amount': 'std'}).reset_index().rename(columns={'Amount': 'monetary_std'})
    
    #concat
    rfm = pd.concat([frequency_merchant["frequency"],monetary_merchant,monetary_merchant_mean["monetary_mean"],monetary_merchant_min["monetary_min"],monetary_merchant_max["monetary_max"],monetary_merchant_std["monetary_std"],recency_merchant['recency']],axis=1)
    
    del frequency_merchant,monetary_merchant,monetary_merchant_mean,monetary_merchant_min,monetary_merchant_max,monetary_merchant_std,recency_merchant
    
    #Re Order dataframe
    df_GMM = rfm[['recency','frequency','monetary','monetary_mean','monetary_min','monetary_max','monetary_std','customer_id']]
    
    del rfm
    #Make it array
    df_GMM_array = df_GMM[['recency','frequency','monetary','monetary_mean','monetary_min','monetary_max','monetary_std']].values
    
    #GaussianMixture Model
    gmm = GaussianMixture(n_components=5,covariance_type='spherical').fit(df_GMM_array)
    labels = gmm.predict(df_GMM_array)
    score = gmm.predict_proba(df_GMM_array)
    df_GMM['labels'] = labels 
    
    #Score adding
    scores = pd.DataFrame(score)
    display(scores)
    scores.rename(columns = {0:'label_zero',1:'label_one',2:'label_two',3:'label_three',4:'label_four'}, inplace = True)

    #Reset Index
    df_GMM = df_GMM.reset_index(drop=True)
    scores = scores.reset_index(drop=True)

    #merge two dataframe
    gmm_score = pd.concat([df_GMM,scores],axis=1)
    
    del df_GMM,scores

    #label_zero
    label_zero = gmm_score.query("labels == 0").sort_values("label_zero").reset_index(drop=True)
    label_zero = label_zero.drop(columns=['label_one','label_two','label_three','label_four'])
    label_zero=label_zero.rename(columns = {'label_zero':'score'})

    #label_one
    label_one = gmm_score.query("labels == 1").sort_values("label_one").reset_index(drop=True)
    label_one = label_one.drop(columns=['label_zero','label_two','label_three','label_four'])
    label_one=label_one.rename(columns = {'label_one':'score'})

    #label_two
    label_two = gmm_score.query("labels == 2").sort_values("label_two").reset_index(drop=True)
    label_two = label_two.drop(columns=['label_zero','label_one','label_three','label_four'])
    label_two=label_two.rename(columns = {'label_two':'score'})


    #label_three
    label_three = gmm_score.query("labels == 3").sort_values("label_three").reset_index(drop=True)
    label_three = label_three.drop(columns=['label_zero','label_one','label_two','label_four'])
    label_three=label_three.rename(columns = {'label_three':'score'})

    #label_four
    label_four = gmm_score.query("labels == 4").sort_values("label_four").reset_index(drop=True)
    label_four = label_four.drop(columns=['label_zero','label_one','label_two','label_three'])
    label_four=label_four.rename(columns = {'label_four':'score'})

    #merging labels
    frames = [label_zero,label_one,label_two,label_three,label_four]
    vertical_stack = pd.concat(frames, axis=0).reset_index(drop=True)
    

    
    #MinMax Scaling for scoring
    vertical_stack_scaling = vertical_stack[['score']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_scores = scaler.fit_transform(vertical_stack_scaling)
    dataset_scaled_scores = pd.DataFrame(scaled_scores)
    
    dataset_scaled_scores.rename(columns = {0:'score'}, inplace = True)
    
    dataset_scaled_scores['score'] = 1 - dataset_scaled_scores['score'] 
    
    #Drop unnecessary columns
    drop_elements = ['score']
    vertical_stack = vertical_stack.drop(drop_elements, axis=1)
    #Merge
    result = pd.concat([dataset_scaled_scores,vertical_stack],axis=1)
    #Reorder df
    del dataset_scaled_scores,vertical_stack
    result = result[['customer_id','recency','frequency','monetary','monetary_mean','monetary_min','monetary_max','monetary_std','labels','score']]
    
    return result