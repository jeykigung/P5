'''
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
Zeroshot Tasks -- 1 Prompt Family (Z)
'''

all_tasks = {}

# =====================================================
# Task Subgroup 1 -- Rating -- 10 Prompts
# =====================================================

task_subgroup_1 = {}

template = {}

'''
Input template:
Which star rating will user {{user_id}} give item {{item_id}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''

template['source'] = "Which star rating will user_{} give item_{} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-1"

task_subgroup_1["1-1"] = template


template = {}
'''
Input template:
How will user {{user_id}} rate this product: {{item_title}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "How will user_{} rate this product : {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-2"

task_subgroup_1["1-2"] = template


template = {}
'''
Input template:
Will user {{user_id}} give item {{item_id}} a {{star_rating}}-star rating? (1 being lowest and 5 being highest)
 
 
Target template:
{{answer_choices[label]}} (yes/no)
 
 
Metrics:
Accuracy
'''
template['source'] = "Will user_{} give item_{} a {}-star rating ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'star_rating']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3"

task_subgroup_1["1-3"] = template


template = {}
'''
Input template:
Does user {{user_id}} like or dislike item {{item_id}}?

 
Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)
 
Metrics:
Accuracy
'''
template['source'] = "Does user_{} like or dislike item_{} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-4"

task_subgroup_1["1-4"] = template


template = {}
'''
Input template:
Predict the user {{user_id}}'s preference on item {{item_id}} ({{item_title}})
-1
-2
-3
-4
-5
 
Target template:
{{answer_choices[star_rating-1]}}
 
Metrics:
Accuracy
'''
template['source'] = "Predict the user_{} 's preference on item_{} ( {} ) \n -1 \n -2 \n -3 \n -4 \n -5"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-5"

task_subgroup_1["1-5"] = template


template = {}

'''
Input template:
What star rating do you think {{user_desc}} will give item {{item_id}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''

template['source'] = "What star rating do you think {} will give item_{} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-6"

task_subgroup_1["1-6"] = template


template = {}
'''
Input template:
How will {{user_desc}} rate this product: {{item_title}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "How will {} rate this product : {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-7"

task_subgroup_1["1-7"] = template


template = {}
'''
Input template:
Will {{user_desc}} give a {{star_rating}}-star rating for {{item_title}}? (1 being lowest and 5 being highest)

 
Target template:
{{answer_choices[label]}} (yes/no)
 
 
Metrics:
Accuracy
'''
template['source'] = "Will {} give a {}-star rating for {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'star_rating', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-8"

task_subgroup_1["1-8"] = template


template = {}
'''
Input template:
Does {{user_desc}} like or dislike {{item_title}}?
 
 
Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)
 
Metrics:
Accuracy
'''
template['source'] = "Does {} like or dislike {} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-9"

task_subgroup_1["1-9"] = template


template = {}
'''
Input template:
Predict {{user_desc}}'s preference towards {{item_title}} (1 being lowest and 5 being highest)
 
Target template:
{{answer_choices[star_rating-1]}}
 
Metrics:
Accuracy
'''
template['source'] = "Predict {} 's preference towards {} ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-10"

task_subgroup_1["1-10"] = template


all_tasks['rating'] =  task_subgroup_1


# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

'''
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the purchase history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me decide ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n try to recommend next item to the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}

'''
Input template:
Given the following purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
predict next possible item for the user
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of {} : \n {} \n predict next possible item for the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}
'''
Input template:
Based on the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you decide the next item likely to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the purchase history of {} : \n {} \n Can you decide the next item likely to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


template = {}
'''
Input template:
Here is the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user?
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history of {} : \n {} \n What to recommend next for the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


# Extractive QA
template = {}
'''
Input template:
Here is the purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
Select the next possible item likely to be purchased by the user from the following candidates:
{{candidate {{item_id}}}}
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history of user_{} : \n {} \n Select the next possible item likely to be purchased by the user from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}
'''
Input template:
Given the following purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user? Select one from the following items:
{{candidate {{item_id}}}}
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Given the following purchase history of {} : \n {} \n What to recommend next for the user? Select one from the following items : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'purchase_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}
'''
Input template:
Based on the purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
Choose the next possible purchased item from the following candidates:
{{candidate {{item_id}}}}
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the purchase history of user_{} : \n {} \n Choose the next possible purchased item from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}
'''
Input template:
I find the purchase history list of {{user_desc}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Try to select one from the following candidates:
{{candidate {{item_id}}}}
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the purchase history list of {} : \n {} \n I wonder which is the next item to recommend to the user . Try to select one from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'purchase_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template


# Pairwise Prediction
template = {}
'''
Input template:
User {{user_id}} has the following purchase history:
{{history item list of {{item_id}}}}
Does the user likely to buy {{item [item_id]}} next?
 
Target template:
{{answer_choices[label]}} (yes/no)
 
Metrics:
Accuracy
'''
template['source'] = "user_{} has the following purchase history : \n {} \n does the user likely to buy {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-11"

task_subgroup_2["2-11"] = template


template = {}
'''
Input template:
According to {{user_desc}}'s purchase history list:
{{history item list of {{item_id}}}}
Predict whether the user will purchase {{item [item_id]}} next?
 
Target template:
{{answer_choices[label]}} (yes/no)
 
Metrics:
Accuracy
'''
template['source'] = "According to {} 's purchase history list : \n {} \n Predict whether the user will purchase {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'purchase_history', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-12"

task_subgroup_2["2-12"] = template


template = {}
'''
Input template:
According to the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you recommend the next possible item to the user?
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "According to the purchase history of {} : \n {} \n Can you recommend the next possible item to the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-13"

task_subgroup_2["2-13"] = template


all_tasks['sequential'] =  task_subgroup_2


# ====================================================
# Task Subgroup 3 -- Explanation -- 12 Prompts
# ====================================================

task_subgroup_3 = {}

template = {}

'''
Input template:
Generate an explanation for user {{user_id}} about this product: {{item_title}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Generate an explanation for user_{} about this product : {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-1"

task_subgroup_3["3-1"] = template


template = {}
'''
Input template:
Given the following review headline 
{{review_headline}}
can you help generate an explanation of user {{user_id}} for item {{item_id}}?


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''
template['source'] = "Given the following review headline \n {} \n can you help generate an explanation of user_{} for item_{} ?"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['review_headline', 'user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-2"

task_subgroup_3["3-2"] = template


template = {}
'''
Input template:
Help user {{user_id}} generate a {{star_rating}}-star explanation about this product: 
{{item_title}}
 
 
Target template:
{{explanation}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Help user_{} generate a {}-star explanation about this product : \n {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'star_rating', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-3"

task_subgroup_3["3-3"] = template


template = {}

'''
Input template:
Generate an explanation for {{user_desc}} about this product: {{item_title}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Generate an explanation for {} about this product : {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-4"

task_subgroup_3["3-4"] = template


template = {}
'''
Input template:
Based on the following review headline:
{{review_headline}}
Generate {{user_desc}}'s purchase explanation about {{item_title}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''
template['source'] = "Based on the following review headline : \n {} \n Generate {} 's purchase explanation about {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['review_headline', 'user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-5"

task_subgroup_3["3-5"] = template


template = {}
'''
Input template:
Help {{user_desc}} generate a {{star_rating}}-star explanation for item {{item_id}}
 
 
Target template:
{{explanation}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Help {} generate a {}-star explanation for item_{}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'star_rating', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-6"

task_subgroup_3["3-6"] = template


template = {}

'''
Input template:
Predict the star rating, then use {{feature}} as feature word to generate user {{user_id}} 's purchase explanation for item {{item_id}}


Target template:
{{star_rating}}, {{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Predict the star rating , then use {} as feature word to generate user_{} 's purchase explanation for item_{}"
template['target'] = "{} , {}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['feature', 'user_id', 'item_id']
template['target_argc'] = 2
template['target_argv'] = ['star_rating', 'explanation']
template['id'] = "3-7"

task_subgroup_3["3-7"] = template


template = {}

'''
Input template:
What score will {{user_desc}} rate item {{item_id}}? Then give an explanation for the rating score. (1 being lowest and 5 being highest)


Target template:
{{star_rating}}, {{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "What score will {} rate item_{} ? Then give an explanation for the rating score . ( 1 being lowest and 5 being highest )"
template['target'] = "{} , {}"
template['task'] = "explanation"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_id']
template['target_argc'] = 2
template['target_argv'] = ['star_rating', 'explanation']
template['id'] = "3-8"

task_subgroup_3["3-8"] = template


template = {}
'''
Name:
Input template:
Based on the feature word {{feature}}, generate an explanation for user {{user_id}} about this product: {{item_title}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Based on the feature word {} , generate an explanation for user_{} about this product : {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['feature', 'user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-9"

task_subgroup_3["3-9"] = template


template = {}
'''
Input template:

Given the word {{feature}}, can you help generate an explanation for {{user_desc}} about the product: \n {{item_title}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Given the word {} , can you help generate an explanation for {} about the product : \n {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['feature', 'user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-10"

task_subgroup_3["3-10"] = template


template = {}
'''
Name:
Input template:
Using the word {{feature}}, write a {{star_rating}}-star explanation for user {{user_id}} about item {{item_id}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Using the word {} , write a {}-star explanation for user_{} about item_{}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['feature', 'star_rating', 'user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-11"

task_subgroup_3["3-11"] = template


template = {}
'''
Name:
Input template:
According to the feature word {{feature}}, generate a {{star_rating}}-star explanation for {{user_desc}} about item {{item_id}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "According to the feature word {} , generate a {}-star explanation for {} about item_{}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['feature', 'star_rating', 'user_desc', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-12"

task_subgroup_3["3-12"] = template


all_tasks['explanation'] = task_subgroup_3


# ====================================================
# Task Subgroup 4 -- Review -- 4 Prompts
# ====================================================

task_subgroup_4 = {}

template = {}

'''
Input template:
Write a short sentence to summarize the following product review from user {{user_id}}:
{{review_body}}
 
 
Target template:
{{review_headline}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Write a short sentence to summarize the following product review from user_{} : \n {}"
template['target'] = "{}"
template['task'] = "review"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'review_body']
template['target_argc'] = 1
template['target_argv'] = ['review_headline']
template['id'] = "4-1"

task_subgroup_4["4-1"] = template


template = {}
'''
Input template:
Given the following review written by user {{user_id}}: 
{{review_body}}
Can you predict the associated star rating (1 being lowest and 5 being highest)?


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "Given the following review written by user_{} : \n {} \n Can you predict the associated star rating ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "review"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'review_body']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "4-2"

task_subgroup_4["4-2"] = template


template = {}
'''
Input template:
Give a short sentence describing the following product review from {{user_desc}}:
{{review_body}}
 
 
Target template:
{{review_headline}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Give a short sentence describing the following product review from {}: \n {}"
template['target'] = "{}"
template['task'] = "review"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'review_body']
template['target_argc'] = 1
template['target_argv'] = ['review_headline']
template['id'] = "4-3"

task_subgroup_4["4-3"] = template


template = {}
'''
Input template:
According to the following review written by {{user_desc}}:
{{review_body}}
Predict the associated star rating (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "According to the following review written by {} : \n {} \n Predict the associated star rating ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "review"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'review_body']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "4-4"

task_subgroup_4["4-4"] = template


all_tasks['review'] = task_subgroup_4


# =====================================================
# Task Subgroup 5 -- Traditional -- 8 Prompts
# =====================================================

task_subgroup_5 = {}

template = {}

'''
Input template:
Will user {{user_id}} likely to interact with item {{item_id}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Will user_{} likely to interact with item_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-1"

task_subgroup_5["5-1"] = template


template = {}

'''
Input template:
Shall we recommend item {{item_id}} to {{user_desc}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Shall we recommend item_{} to {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['item_id', 'user_desc']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-2"

task_subgroup_5["5-2"] = template


template = {}

'''
Input template:
For {{user_desc}}, do you think it is good to recommend {{item_title}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "For {}, do you think it is good to recommend {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-3"

task_subgroup_5["5-3"] = template


template = {}

'''
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "I would like to recommend some items for user_{} . Is the following item a good choice ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-4"

task_subgroup_5["5-4"] = template


template = {}

'''
Input template:
Which item of the following to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Which item of the following to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-5"

task_subgroup_5["5-5"] = template


template = {}

'''
Input template:
Choose the best item from the candidates to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Choose the best item from the candidates to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-6"

task_subgroup_5["5-6"] = template


template = {}

'''
Input template:
Pick the most suitable item from the following list and recommend to user {{user_id}}:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Pick the most suitable item from the following list and recommend to user_{} : \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-7"

task_subgroup_5["5-7"] = template


template = {}

'''
Input template:
We want to make recommendation for user {{user_id}}. Select the best item from these candidates:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "We want to make recommendation for user_{} .  Select the best item from these candidates : \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-8"

task_subgroup_5["5-8"] = template


all_tasks['traditional'] = task_subgroup_5


# ========================================================
# Cold-Start/Zero-Shot Task Subgroup - 7 Prompts
# ========================================================

'''
Zero-Shot Inference Tasks
'''

zero_short_tasks = {}

template = {}

'''
Input template:
Given the facts about the new product, do you think user {{user_id}} will like or dislike it?
title: {{item_title}}
brand: {{brand}}
price: {{price}}


Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)


Metrics:
Accuracy
'''

template['source'] = "Given the facts about the new product , do you think user_{} will like or dislike it ? \n title : {} \n brand : {} \n price : {}"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'item_title', 'brand', 'price']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "Z-1"

zero_short_tasks["Z-1"] = template


template = {}

'''
Input template:
Here are the details about a new product: 
title: {{item_title}}
brand: {{brand}}
price: {{price}}
What star will {{user_desc}} probably rate the product?
-1
-2
-3
-4
-5

Target template:
{{answer_choices[star_rating-1]}}


Metrics:
Accuracy
'''

template['source'] = "Here are the details about a new product : \n title : {} \n brand : {} \n price : {} \n What star will {} probably rate the product ? \n -1 \n -2 \n -3 \n -4 \n -5"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 4
template['source_argv'] = ['item_title', 'brand', 'price', 'user_desc']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "Z-2"

zero_short_tasks["Z-2"] = template


template = {}

'''
Input template:
Predict user {{user_id}}'s preference about the new product (1 being lowest and 5 being highest):
title: {{item_title}}
price: {{price}}
brand: {{brand}}


Target template:
{{answer_choices[star_rating-1]}}


Metrics:
Accuracy
'''

template['source'] = "Predict user_{} 's preference about the new product ( 1 being lowest and 5 being highest ) : \n title : {} \n price : {} \n brand : {}"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'item_title', 'price', 'brand']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "Z-3"

zero_short_tasks["Z-3"] = template


template = {}

'''
Input template:
Will {{user_desc}} like or dislike the following product?
title: {{item_title}}
price: {{price}}
brand: {{brand}}

Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)


Metrics:
Accuracy
'''

template['source'] = "Will {} like or dislike the following product ? \n title : {} \n price : {} \n brand : {}"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 4
template['source_argv'] = ['user_desc', 'item_title', 'price', 'brand']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "Z-4"

zero_short_tasks["Z-4"] = template


template = {}

'''
Input template:
Generate a possible explanation for {{user_desc}}'s preference about the following product: 
title: {{item_title}}
brand: {{brand}}
price: {{price}}

Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Generate a possible explanation for {} 's preference about the following product : \n title : {} \n brand : {} \n price : {}"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 4
template['source_argv'] = ['user_desc', 'item_title', 'brand', 'price']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "Z-5"

zero_short_tasks["Z-5"] = template


template = {}

'''
Input template:
Based on the word {{feature}}, help user {{user_id}} write a {{star_rating}}-star explanation for this new product: 
title: {{item_title}}
price: {{price}}
brand: {{brand}}

Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Based on the word {} , help user_{} write a {}-star explanation for this new product : \n title : {} \n price : {} \n brand : {}"
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 6
template['source_argv'] = ['feature', 'user_id', 'star_rating', 'item_title', 'price', 'brand']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "Z-6"

zero_short_tasks["Z-6"] = template


template = {}

'''
Input template:
For the new product {{item_title}}, we would like to know whether {{user_desc}} will love it. If you think the user will love it, please help explain why.


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "For the new product {} , we would like to know whether {} will love it . If you think the user will love it , please help explain why ."
template['target'] = "{}"
template['task'] = "zeroshot"
template['source_argc'] = 2
template['source_argv'] = ['item_title', 'user_desc']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "Z-7"

zero_short_tasks["Z-7"] = template
