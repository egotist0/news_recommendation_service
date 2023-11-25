# News Recommendation Tool

Link: https://github.com/egotist0/EvaDB_Proj

[toc]



## Introduction

EVA is an open-source AI-relational database with support for deep learning models. It aims to support AI-powered database applications that operate on both structured and unstructured data using deep learning models. The database has built-in support for popular vector databases like Pinecone.

Many content platforms thrive on suggesting related content to their users. The more relevant items the platform can provide, the longer the user will stay on the site, which often translates to increased ad revenue for the company. For example, a platform might implement a tag-based recommendation engine: if you read a "Business" article, it suggests five more articles tagged as "Business." However, an even better approach to building a recommendation engine is to use similarity search and a machine learning algorithm.

The project aims to develop an article recommendation tool. It utilizes EvaDB, ChatGPT models, and Pinecone for semantic similarity matching to provide functionalities such as document summarization, keyword extraction, and entity recognition for database documents. This tool can select the top 10 articles from the article library that are most likely to align with the reader's reading preferences based on their previous reading history and provide the recommended results.

## Data Sources:

The data for the articles is sourced from https://www.kaggle.com/datasets/snapcrack/all-the-news. It includes 143,000 articles from 15 American publications, including the New York Times, Breitbart, CNN, Business Insider, the Atlantic, Fox News, Talking Points Memo, Buzzfeed News, National Review, New York Post, the Guardian, NPR, Reuters, Vox, and the Washington Post.

This project exclusively utilizes the "articles1.csv" file from the dataset, which comprises 50,000 news articles (Articles 1-50,000). The file includes the following attributes:

![image-20231017225830066](https://github.com/egotist0/EvaDB_Proj/tree/master/Photo/photo1.png)



## Related Work

### Pinecone

Pinecone is an emerging service and tool designed to assist organizations and developers in effectively managing and leveraging large-scale vector data. It is a high-performance vector indexing and retrieval system specifically developed for machine learning applications.

Pinecone provides a powerful infrastructure for storing, indexing, and searching vector embeddings. Vector embeddings are numerical representations of data points that capture their semantic information and relationships. These embeddings are widely used in various domains, including natural language processing, computer vision, recommendation systems, and anomaly detection. One of the key advantages of Pinecone is its ability to handle high-dimensional vector data efficiently. It employs advanced indexing techniques, such as approximate nearest neighbor search algorithms, to enable fast and accurate retrieval of similar vectors. This capability is particularly valuable in scenarios where real-time or near-real-time responses are required, such as personalized recommendations or similarity-based search.



### GloVe

Global Vectors for Word Representation (GloVe) is a popular word embedding model developed by Stanford University researchers. It represents words as dense vectors in a high-dimensional space, capturing their semantic relationships based on co-occurrence patterns. GloVe combines global statistical information with local context to generate word vectors. By factorizing a co-occurrence matrix, GloVe produces word vectors where the dot product represents the likelihood of word co-occurrence. It excels in capturing both syntactic and semantic relationships between words, making it valuable for natural language processing tasks like word similarity computation, text classification, and machine translation. GloVe is known for its simplicity, efficiency, and effectiveness, making it widely adopted in academia and industry. Pre-trained GloVe word vectors are available for multiple languages and easily integrated into machine learning models, empowering advancements in language understanding and text analysis.



## Technical Details

1. Data Preprocessing

   Use Python's Pandas library to process data in CSV format. The code is as follows:

   ```python
   data = pd.read_csv("articles.csv")
   
   # prepare the data
   # rename id column and remove unnecessary columns
   data.rename(columns={"Unnamed: 0": "article_id"}, inplace = True)
   data.drop(columns=['date'], inplace = True)
   
   # extract only first few sentences of each article for quicker vector calculations
   data['content'] = data['content'].fillna('')
   data['content'] = data.content.swifter.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:4]))
   data['title_and_content'] = data['title'] + ' ' + data['content']
   ```

2. Build Vector Index

   I create a vector index using the Title and Content attributes of each article. The dimension of the vectors is set to 300, and the metric is set to "cosine". I build this index in Pinecone to facilitate subsequent similarity searches on the entire content and titles of the article database.The vector creation model used is GloVe.

   ```pYTHON
   pinecone.init(api_key=api_key, environment=environment)
   if PINECONE_INDEX_NAME in pinecone.list_indexes():
           pinecone.delete_index(PINECONE_INDEX_NAME)
   pinecone.create_index(dimension=300, name=PINECONE_INDEX_NAME, metric="cosine", shards=1)
   pinecone_index = pinecone.Index(index_name=PINECONE_INDEX_NAME)
   
   model = SentenceTransformer('average_word_embeddings_komninos')
   
   # create a vector embedding based on title and article columns
   encoded_articles = model.encode(data['title_and_content'], show_progress_bar=True)
   data['article_vector'] = pd.Series(encoded_articles.tolist())
   
   
   # upload the data into pinecone
   upsert_batch = []
   for i, row in data.iterrows():
       upsert_batch.append((str(row.id), row.article_vector))
   
       if len(upsert_batch) > 500:
           pinecone_index.upsert(upsert_batch)
           upsert_batch = []
   
   # Process any remaining data in upsert_batch
   if upsert_batch:
       pinecone_index.upsert(upsert_batch)
   ```

3. Get recommended articles

   Pinecone will retrieve the articles that are closest to a given object vector based on the similarity of the index. Since the reader may have read multiple articles, the source points for similarity search are multiple vectors. As the `Similarity` function in EvaDB does not provide the functionality to search for the similarity of multiple objects, I have opted to use the `query` API provided by Pinecone itself.

   ```Python
   def article_recomendation(artile_list):
       meta_list = data.loc[data['id'].isin(artile_list)]
   
       article_vectors = meta_list['article_vector']
       article_vector = [*map(mean, zip(*article_vectors))]
   
       query_results = pinecone_index.query(vector=article_vector, top_k=10)
       res = query_results['matches']
   
       results_list = []
   
       for idx, item in enumerate(res):
           results_list.append({
               "title": titles_mapped[int(item.id)],
               "publication": publications_mapped[int(item.id)],
               "content": content_mapped[int(item.id)],
           })
   
       return results_list
   ```

4. Providing website services

   This service can be further conceptualized as a web application built using [Django REST framework](https://www.django-rest-framework.org/). However, due to time constraints, this step is still a work in progress.



## Sample Input/Output

+ Input:

  Id_list: [50812, 51046, 51048, 51095, 54559]

  Each ID represents the number of the article in the database. This list can have any number of members.

+ Output:

  Recommendation_article:

  ```html
  [{'title': 'New cervical cancer research is personal ',
    'publication': 'CNN',
    'content': ' (CNN) A new study published earlier this week in the journal Cancer reveals that mortality rates are far higher and racial disparities in mortality are far larger than was previously thought for cervical cancer,  a disease that can be screened for and for which there is a vaccine.  As an Osteopathic Family Physician who provides women’s health services to my patients as well as medical care for general medical conditions, these findings directly impact my patients and my profession. As a black woman, they affect me personally. As a family physician, this study was a hard pill to swallow, but as a black woman it is even harder.'},
   .
   .
   .
   {'title': 'Gene Tests Identify Breast Cancer Patients Who Can Skip Chemotherapy, Study Says - The New York Times',
    'publication': 'New York Times',
    'content': 'When is it safe for a woman with breast cancer to skip chemotherapy? A new study helps answer that question, based on a test of gene activity in tumors. It found that nearly half of women with   breast cancer who would traditionally receive chemo can avoid it, with little risk of the cancer coming back or spreading in the next five years. The   genomic test measures the activity of genes that control the growth and spread of cancer, and can identify women with a low risk of recurrence and therefore little to gain from chemo. “More and more evidence is mounting that there is a substantial number of women with breast cancer who will not need chemotherapy to do well,” said Dr.'}]
  ```

  Includes 10 articles that the current reader may find most interesting, including the corresponding article titles, publications, and content.

  

## Testing Approach:

Using the EvaDB ChatGPT API interface, input the titles and contents of the articles the reader has previously read, as well as the titles and contents of ten recommended articles. Then, let ChatGPT make judgments from a semantic perspective.

+ Test 1

  ```python
  # Person whose interest lies in cancer and health
  health_title_df = data[data['title'].str.contains('cancer')].head(5)
  # get id list
  id_list = health_title_df['id'].tolist()
  
  # get title list
  original_articles = []
  for idx, item in enumerate(id_list):
    original_articles.append({
        "title": titles_mapped[int(item)],
        "publication": publications_mapped[int(item)],
        "content": content_mapped[int(item)],
        })
  
  recommendation_articles = article_recomendation(id_list)
  
  # Evaluation
  content = "I will provide the titles and contents of five articles I have read before, as well as 10 articles recommended to me. Please evaluate the similarity in themes and relevance of these 10 articles to my previously read articles.\n"
  original = "history articles: "+str(original_articles) + "\n"
  recommendation = "recommendation articles:"+str(recommendation_articles)
  
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                         messages=[{"role": "user", "content": content + original + recommendation}])
  
  # print the completion
  print(completion.choices[0].message.content)
  ```

  Evaluation result:

  The themes of the previously read articles are focused on various types of cancer, including breast cancer, cervical cancer, and skin cancer. They also highlight issues such as cancer mortality rates, racial disparities in cancer deaths, and the use of AI technology in cancer diagnosis. 

  The recommended articles cover similar themes, with a focus on cancer research and prevention. They discuss topics such as the impact of a high-fat diet on prostate cancer, the effectiveness of ovarian cancer prevention surgery, the racial gap in cervical cancer deaths, the use of AI in detecting skin cancer, and the ability to identify breast cancer patients who can skip chemotherapy through gene testing. 

  Overall, the recommended articles are highly relevant to the previously read articles, as they all revolve around cancer research, prevention, and treatment. The themes explored in the recommended articles align closely with the themes of the previously read articles, indicating a similarity of interests and relevance in the topics discussed.

+ Test 2

  ```python
  # Person whose interest lies in computer science
  health_title_df = data[data['title'].str.contains('Software Engineer')].head(5)
  # get id list
  id_list = health_title_df['id'].tolist()
  
  # get title list
  original_articles = []
  for idx, item in enumerate(id_list):
    original_articles.append({
        "title": titles_mapped[int(item)],
        "publication": publications_mapped[int(item)],
        "content": content_mapped[int(item)],
        })
  
  recommendation_articles = article_recomendation(id_list)
  
  # Evaluation
  content = "I will provide the titles and contents of five articles I have read before, as well as 10 articles recommended to me. Please evaluate the similarity in themes and relevance of these 10 articles to my previously read articles.\n"
  original = "history articles: "+str(original_articles) + "\n"
  recommendation = "recommendation articles:"+str(recommendation_articles)
  
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                         messages=[{"role": "user", "content": content + original + recommendation}])
  
  # print the completion
  print(completion.choices[0].message.content)
  
  ```

  Evaluation result:

  The previously read article is about a report stating that only 36 percent of Indian software engineers can write useful code. The article discusses the findings of a study conducted by Aspiring Minds, an Indian skills assessment company, which assessed the programming skills of Indian software engineers using an automated tool called Automata. The study found that only a low percentage of engineers were able to write working, compilable code.

  The recommended articles cover a range of topics, but there are several that are related to technology, programming, and education. One article discusses Apple's free coding education app called Swift Playgrounds, which aims to teach children coding using the Swift programming language. Another article explores the topic of online dating and how it can lead to the lowering of standards. There are also articles about Apple's investments in artificial intelligence and a study that claims children are more proficient in operating smartphones than completing basic tasks. Additionally, there are articles about an education startup called Newsela and its impact on literacy in American classrooms, leadership lessons from a book recommended by Facebook's HR chief, and a compilation of the best websites for learning something new.

  Overall, while not all of the recommended articles directly relate to the theme of the previously read article, there are several that touch on technology, programming, and education, which are relevant themes.



The above two test cases reveal that there is a high correlation between the recommended articles and the reader's previous reading records.

## Lessons Learned

I have primarily learned how to use the vector database Pinecone and understand its working principles, as well as its relevant use cases. Additionally, I have conducted some research on GloVe. The usage of EvaDB has also helped me solidify my understanding of SQL syntax and the strong correlation between AI and vector databases.

## Challenges Faced 

1. Pinecone's indexing speed can be slow because all service requests are made through the API and over the network to the Pinecone platform. Consequently, when dealing with a large amount of data, the indexing process can be time-consuming. To address this, I adopted the Batch Process approach, where I index 500 data points at a time, effectively speeding up the progress.
2. For similarity search based on multiple query points, EvaDB does not provide this functionality. Therefore, after reviewing the EvaDB documentation extensively, I decided to use Pinecone's Query API to accomplish this task.



This experiment was conducted on the Google Colab platform.

## Reference

+ https://www.kaggle.com/datasets/snapcrack/all-the-news/data
+ https://docs.pinecone.io/docs/choosing-index-type-and-size
+ https://nlp.stanford.edu/projects/glove/?ref=hackernoon.com
+ https://github.com/thawkin3/pinecone-demo
+ https://evadb.readthedocs.io/en/latest/source/reference/evaql/create_table.html
